# Outcome (Y) randomisation + Mirror Statistic (from DS)

module wrapper_pipeline_inference

    using GLM
    using GLMNet
    using Distributions
    using Random
    using StatsPlots
    using Plots
    using LinearAlgebra
    using DataFrames

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
            beta_pool=[-1., -0.8, -0.5, 0.5, 0.8, 1.],
            beta_intercept=beta_intercept,
            sigma2=sigma2
        )

        " Estimate via Randomisation "
        # randomisation of Y
        # U = Y + W
        # W ~ N(0, gamma*Sigma), gamma > 0

        u, v = randomisation_ds.randomisation(y=data.y, X=data.X, gamma=gamma_randomisation, estimate_sigma2=estimate_sigma2, sigma2=sigma2)

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
        

        " Mirror Statistic on single data splitting "
        ds_selection = mirror_statistic.ds(X=data.X, y=data.y, fdr_level=fdr_level)

        # FDR MS
        fdr_ds_ms = classification_metrics.false_discovery_rate(
            true_coef=data.beta_true .!= 0,
            estimated_coef=ds_selection
        )

        # TPR MS
        tpr_ds_ms = classification_metrics.true_positive_rate(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=ds_selection
        )

        # Power
        power_ds_ms = classification_metrics.power(
            true_coef=data.beta_true .!= 0.,
            estimated_coef=ds_selection
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


    """
        Run inference on data with the provided methods
    """
    function wrapper_inference(;
        data,
        estimate_sigma2=true,
        methods_to_evaluate=["DS", "MDS", "Rand_MS"],
        fdr_level=0.1
        )

        metrics_array = []

        if "Rand_MS" in methods_to_evaluate
            rand_ms_selection = randomisation_ds.rand_ms(
                y=data.y,
                X=data.X,
                sigma2=data.sigma2,
                gamma=1.,
                estimate_sigma2=estimate_sigma2,
                fdr_level=fdr_level
            )

            push!(
                metrics_array,
                classification_metrics.wrapper_metrics(data.beta_true .!= 0, rand_ms_selection)
            )
        end

        if "DS" in methods_to_evaluate
            ds_selection = mirror_statistic.ds(X=data.X, y=data.y, fdr_level=fdr_level)

            push!(
                metrics_array,
                classification_metrics.wrapper_metrics(data.beta_true .!= 0, ds_selection)
            )
        end

        if "MDS" in methods_to_evaluate
            mds_selection = mirror_statistic.mds(
                X=data.X,
                y=data.y,
                n_ds=50,
                fdr_level=fdr_level
            )

            push!(
                metrics_array,
                classification_metrics.wrapper_metrics(data.beta_true .!= 0, mds_selection)
            )
        end

        return metrics_array

    end

end
