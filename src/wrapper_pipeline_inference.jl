# Outcome (Y) randomisation + Mirror Statistic (from DS)

module wrapper_pipeline_inference

    using GLM
    using GLMNet
    using Distributions
    using Random
    using StatsPlots
    using Plots
    using LinearAlgebra

    include("./utilities/data_generation.jl")
    include("./utilities/randomisation_ds.jl")
    include("./utilities/mirror_statistic.jl")
    include("./utilities/classification_metrics.jl")


    """
        lasso_plus_ols(
            X1::AbstractArray,
            X2::AbstractArray,
            y1::Vector{Float64},
            y2::Vector{Float64},
            add_intercept=true
            )

        Estimate LASSO + OLS on the splitted data (either randomisation or standard data splitting)
    """
    function lasso_plus_ols(;
        X1::AbstractArray,
        X2::AbstractArray,
        y1::Vector{Float64},
        y2::Vector{Float64},
        add_intercept=true
    )
        # Lasso from GLMNet includes the intercept by default
        lasso_cv = GLMNet.glmnetcv(X1, y1)
        lasso_coef = GLMNet.coef(lasso_cv)
        # pushfirst!(lasso_coef, lasso_cv.path.a0[argmin(lasso_cv.meanloss)])

        # Non-0 coefficients
        non_zero = lasso_coef .!= 0

        # Now LM(OLS) estimate on the second half of the data
        # either from randomisation: V = Y - 1/gamma*W ~ N(mu, sigma2(1 + 1/gamma)I_n)
        # or from Data Splitting
        # GLM in Matrix notation does NOT include the intercept by default
        X2 = X2[:, non_zero]
        p = size(X2)[2]
        if add_intercept
            X2 = hcat(X2, ones(size(X2)[1], 1))
            p += 1
        end
        lm_on_split2 = GLM.lm(X2, y2)

        if add_intercept
            lm_on_v_coef = GLM.coef(lm_on_split2)[1:(p - 1)]
            lm_pvalues_subset = GLM.coeftable(lm_on_split2).cols[4][1:(p - 1)]
            lm_coef_int = last(GLM.coef(lm_on_split2))
        else
            lm_on_v_coef = GLM.coef(lm_on_split2)
            lm_pvalues_subset = GLM.coeftable(lm_on_split2).cols[4]
            lm_coef_int = 0.
        end

        lm_coef = zeros(length(lasso_coef))
        lm_pvalues = ones(length(lasso_coef)) # pvalue for excluded vars assumed to be 1
        lm_coef[non_zero] = lm_on_v_coef
        lm_pvalues[non_zero] = lm_pvalues_subset

        return (lasso_coef=lasso_coef, lm_coef=lm_coef, lm_pvalues=lm_pvalues)
    end

    """
        Wrapper to run the whole pipeline, data generation, randomisation inference and FDR calculation
    """
    function wrapper_randomisation_inference(;
        n,
        p,
        correlation_coefficients,
        prop_zero_coef,
        sigma2=1.,
        gamma_randomisation=1.,
        fdr_level=0.1
        )

        data = data_generation.linear_regression_data(
            n=n,
            p=p,
            correlation_coefficients=correlation_coefficients,
            prop_zero_coef=prop_zero_coef,
            beta_intercept=1.,
            sigma2=sigma2
        )

        " Estimate via Randomisation "
        # randomisation of Y
        # U = Y + W
        # W ~ N(0, gamma*Sigma), gamma > 0
        u, v = randomisation_ds.randomisation(y=data.y, gamma=gamma_randomisation, sigma2=sigma2)

        " Perform variable selection on U using Lasso and Inference on V using OLS "
        lasso_coef, lm_coef, lm_pvalues = lasso_plus_ols(
            X1=data.X,
            X2=data.X,
            y1=u,
            y2=v,
            add_intercept=true
        )
        
        # Check FDR with just the coefficients from LM on randomised data
        fdr_rand_raw = classification_metrics.false_discovery_rate(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=lm_pvalues .<= 0.05
        )

        # TPR rand
        tpr_rand_raw = classification_metrics.true_positive_rate(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=lm_pvalues .<= 0.05
        )

        # # To properly check FDR we need to use the estimated pvalues from the LM regression
        # # and adjust them for the desired FDR level
        # adjusted_pvalues = classification_metrics.bh_correction(p_values=lm_pvalues, fdr_level=fdr_level)

        # fdr_randomisation_bh = classification_metrics.false_discovery_rate(
        #     true_coef=data.beta_true .!= 0,
        #     estimated_coef=adjusted_pvalues[:, 3]
        # )

        " Add Mirror Statistic on top of randomisation "
        ms_coef = mirror_statistic.mirror_stat(lm_coef, lasso_coef)
        # get FDR threshold
        optimal_t = mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

        # FDR Rand + MS
        fdr_rand_ms = classification_metrics.false_discovery_rate(
            true_coef=data.beta_true .!= 0,
            estimated_coef=ms_coef .> optimal_t
        )

        # TPR Rand + MS
        tpr_rand_ms = classification_metrics.true_positive_rate(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=ms_coef .> optimal_t
        )

        " Mirror Statistic on simple data splitting "
        split_data = mirror_statistic.data_splitting(data.X, data.y)

        lasso_coef, lm_coef, lm_pvalues = lasso_plus_ols(
            X1=split_data.X1,
            X2=split_data.X2,
            y1=split_data.y1,
            y2=split_data.y2,
            add_intercept=true
        )

        # Add Mirror Statistic
        ms_coef = mirror_statistic.mirror_stat(lm_coef, lasso_coef)
        # get FDR threshold
        optimal_t = mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

        # FDR MS
        fdr_ds_ms = classification_metrics.false_discovery_rate(
            true_coef=data.beta_true .!= 0,
            estimated_coef=ms_coef .> optimal_t
        )

        # TPR MS
        tpr_ds_ms = classification_metrics.true_positive_rate(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=ms_coef .> optimal_t
        )

        class_metrics = (
            FDR_rand_plus_MS=fdr_rand_ms,
            FDR_rand_only=fdr_rand_raw,
            FDR_MS_only=fdr_ds_ms,
            TPR_rand_plus_MS=tpr_rand_ms,
            TPR_rand_only=tpr_rand_raw,
            TPR_MS_only=tpr_ds_ms
        )

        return (
            data=data,
            lasso_coef=lasso_coef,
            lm_coef=lm_coef,
            lm_pvalues=lm_pvalues,
            class_metrics=class_metrics
        )

    end

end
