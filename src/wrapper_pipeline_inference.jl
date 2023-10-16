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
    include("./utilities/variable_selection_plus_inference.jl")


    """
        Wrapper to run the whole pipeline, data generation, randomisation inference and FDR calculation
    """
    function wrapper_randomisation_inference(;
        n,
        p,
        correlation_coefficients,
        block_covariance,
        prop_zero_coef,
        sigma2=1.,
        beta_intercept=1.,
        gamma_randomisation=1.,
        estimate_sigma2=true,
        fdr_level=0.1
        )

        data = data_generation.linear_regression_data(
            n=n,
            p=p,
            correlation_coefficients=correlation_coefficients,
            block_covariance=block_covariance,
            prop_zero_coef=prop_zero_coef,
            beta_intercept=beta_intercept,
            sigma2=sigma2
        )

        " Estimate via Randomisation "
        # randomisation of Y
        # U = Y + W
        # W ~ N(0, gamma*Sigma), gamma > 0

        sigma2_estimate = sigma2
        if estimate_sigma2
            # From standard fit for  p << n or from Lasso rss for high-dimensional data
            n_coefficients = p + ifelse(beta_intercept == 0, 0, 1)
            if p < n/4
                sigma2_estimate = GLM.deviance(GLM.lm(data.X, data.y)) / (n - n_coefficients)
            else
                lasso_cv = GLMNet.glmnetcv(data.X, data.y)
                non_zero = sum(GLMNet.coef(lasso_cv) .!= 0)
                yhat = GLMNet.predict(lasso_cv, data.X)

                if non_zero >= (n-1)
                    throw(error("# non-zero coeff > n: impossible to estimate sigma^2"))
                end
                sigma2_estimate = sum((data.y - yhat).^2) / (n - non_zero - 1)

            end
        end

        u, v = randomisation_ds.randomisation(y=data.y, gamma=gamma_randomisation, sigma2=sigma2_estimate)

        " Perform variable selection on U using Lasso and Inference on V using OLS "
        lasso_coef, lm_coef, lm_pvalues = variable_selection_plus_inference.lasso_plus_ols(
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

        # Power
        power_rand_raw = classification_metrics.power(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=lm_pvalues .<= 0.05
        )

        # # To properly check FDR we should use the estimated pvalues from the LM regression
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

        # Power
        power_rand_ms = classification_metrics.power(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=ms_coef .> optimal_t
        )
        

        " Mirror Statistic on simple data splitting "
        split_data = mirror_statistic.data_splitting(data.X, data.y)

        lasso_coef, lm_coef, lm_pvalues = variable_selection_plus_inference.lasso_plus_ols(
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

        # Power
        power_ds_ms = classification_metrics.power(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=ms_coef .> optimal_t
        )
        

        " Multiple Data Splitting (from the Mirror Statistic paper) "
        mds_selection = mirror_statistic.mds(
            X=data.X,
            y=data.y,
            n_ds=50,
            fdr_level=fdr_level
        )

        # FDR MDS
        fdr_mds_ms = classification_metrics.false_discovery_rate(
            true_coef=data.beta_true .!= 0,
            estimated_coef=mds_selection
        )

        # TPR MDS
        tpr_mds_ms = classification_metrics.true_positive_rate(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=mds_selection
        )

        # Power
        power_mds_ms = classification_metrics.power(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=mds_selection
        )
        
        class_metrics = (
            FDR_rand_plus_MS=fdr_rand_ms,
            FDR_rand_only=fdr_rand_raw,
            FDR_MS_only=fdr_ds_ms,
            FDR_MDS_only=fdr_mds_ms,
            TPR_rand_plus_MS=tpr_rand_ms,
            TPR_rand_only=tpr_rand_raw,
            TPR_MS_only=tpr_ds_ms,
            TPR_MDS_only=tpr_mds_ms,
            POWER_rand_plus_MS=power_rand_ms,
            POWER_rand_only=power_rand_raw,
            POWER_MS_only=power_ds_ms,
            POWER_MDS_only=power_mds_ms
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
