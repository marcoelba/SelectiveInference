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


    # Wrapper for data simulation and inference pipeline

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
        # true_zero_coef = findall(data.beta_true .== 0)

        # randomisation of Y
        # U = Y + W
        # W ~ N(0, gamma*Sigma), gamma > 0
        u, v = randomisation_ds.randomisation(y=data.y, gamma=gamma_randomisation, sigma2=sigma2)

        " Perform variable selection on U using Lasso and Inference on V using OLS "
        lasso_coef, lm_coef, lm_pvalues = randomisation_ds.lasso_plus_ols(
            X=data.X,
            u=u,
            v=v,
            add_intercept=true
        )

        # Check FDR with just the coefficients from LM on randomised data
        fdr_randomisation_raw = classification_metrics.false_discovery_rate(
            true_coef=data.beta_true,
            estimate_coef=lm_pvalues .<= 0.05
        )

        # To properly check FDR we need to use the estimated pvalues from the LM regression
        # and adjust them for the desired FDR level
        adjusted_pvalues = classification_metrics.bh_correction(p_values=lm_pvalues, fdr_level=fdr_level)

        fdr_randomisation_bh = classification_metrics.false_discovery_rate(
            true_coef=data.beta_true,
            estimate_coef=adjusted_pvalues[:, 3]
        )

        " Add Mirror Statistic on top of randomisation "
        ms_coef = mirror_statistic.mirror_stat(lm_coef, lasso_coef)
        # get FDR threshold
        optimal_t = mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

        fdr_mirror_statistic = classification_metrics.false_discovery_rate(
            true_coef=data.beta_true,
            estimate_coef=ms_coef .> optimal_t
        )

        return (
            data=data,
            lasso_coef=lasso_coef,
            lm_coef=lm_coef,
            lm_pvalues=lm_pvalues,
            fdr_mirror_statistic=fdr_mirror_statistic,
            fdr_randomisation_bh=fdr_randomisation_bh,
            fdr_randomisation_raw=fdr_randomisation_raw
        )

    end

end
