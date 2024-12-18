# Randomisation technique for variable selection

module randomisation_ds
    using Distributions
    using Random
    using GLM
    using GLMNet
    using LinearAlgebra

    using RandMirror: variable_selection_plus_inference, mirror_statistic

    
    """
        randomisation(;y, gamma, sigma2)
        
        Randomise the outcome Y with an added Gaussian noise of mean 0 and variance proportional to gamma * sigma2,
        where sigma2 is the variance of y

        Return (u = y + w, v = y - 1/gamma*w)
    """
    function randomisation(;y::Vector{Float64}, X::Union{Matrix{Float64}, Transpose{Float64, Matrix{Float64}}}, gamma::Float64, estimate_sigma2::Bool, sigma2::Float64)
        if estimate_sigma2
            n = length(y)
            p = size(X)[2]
            # From standard fit for  p << n or from Lasso rss for high-dimensional data
            # n_coefficients = p + ifelse(beta_intercept == 0, 0, 1)
            n_coefficients = p
            if p < n/4
                sigma2_estimate = GLM.deviance(GLM.lm(X, y)) / (n - n_coefficients)
            else
                max_df = n - Int(round(max(n*0.1, 1)))
                lasso_cv = GLMNet.glmnetcv(X, y, dfmax=max_df)
                non_zero = sum(GLMNet.coef(lasso_cv) .!= 0)
                yhat = GLMNet.predict(lasso_cv, X)

                if non_zero >= (n-1)
                    throw(error("# non-zero coeff > n: impossible to estimate sigma^2"))
                end
                sigma2_estimate = sum((y - yhat).^2) / (n - non_zero - 1)

            end
        else
            sigma2_estimate = sigma2
        end

        n = length(y)
        w = Random.rand(Distributions.Normal(0, sqrt(gamma * sigma2_estimate)), n)
        u = y + w

        v = y - 1/gamma * w

        return (u=u, v=v, sigma2_estimate=sigma2_estimate)
    end

    """
        Selection from Randomisation + Mirror Statistic
    """
    function rand_ms(;
        y::Vector{Float64},
        X::Union{Matrix{Float64}, Transpose{Float64, Matrix{Float64}}},
        sigma2::Float64,
        gamma::Float64=1.,
        estimate_sigma2::Bool=true,
        fdr_level::Float64=0.1,
        alpha_lasso::Float64=1.
        )
        # Do Randomisation
        u, v, sigma2_est = randomisation(
            y=y,
            X=X,
            gamma=gamma,
            estimate_sigma2=estimate_sigma2,
            sigma2=sigma2
        )

        " Perform variable selection on U using Lasso and Inference on V using OLS "
        lasso_coef, lm_coef, lm_pvalues, lm_coef_int, lm_sdterr, lm_dof = variable_selection_plus_inference.lasso_plus_ols(
            X1=X,
            X2=X,
            y1=u,
            y2=v,
            add_intercept=true,
            alpha_lasso=alpha_lasso
        )

        " Add Mirror Statistic on top of randomisation "
        ms_coef = mirror_statistic.mirror_stat(lm_coef, lasso_coef)
        # get FDR threshold
        optimal_t = mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)
        return ms_coef .> optimal_t
        
    end


    """
        real_data_rand_ms

        Randomisation + Mirror Statistic
            for real data analysis
    """
    function real_data_rand_ms(;
        y::Vector{Float64},
        X::Union{Matrix{Float64}, Transpose{Float64, Matrix{Float64}}},
        gamma::Float64=1.,
        fdr_level::Float64=0.1,
        alpha_lasso::Float64=1.
        )
        # Do Randomisation
        u, v, sigma2_est = randomisation(
            y=y,
            X=X,
            gamma=gamma,
            estimate_sigma2=true,
            sigma2=1.
        )

        " Perform variable selection on U using Lasso and Inference on V using OLS "
        lasso_coef, lm_coef, lm_pvalues, lm_coef_int, lm_sdterr, lm_dof = variable_selection_plus_inference.lasso_plus_ols(
            X1=X,
            X2=X,
            y1=u,
            y2=v,
            add_intercept=true,
            alpha_lasso=alpha_lasso
        )

        " Add Mirror Statistic on top of randomisation "
        ms_coef = mirror_statistic.mirror_stat(lm_coef, lasso_coef)
        # get FDR threshold
        optimal_t = mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

        # Return dictionary with all useful objects
        out_dict = Dict(
            ("ms_coef" => ms_coef),
            ("optimal_t" => optimal_t),
            ("selected_ms_coef" => ms_coef .> optimal_t),
            ("lm_coef" => lm_coef),
            ("lm_pvalues" => lm_pvalues),
            ("lm_coef_int" => lm_coef_int),
            ("lm_sdterr" => lm_sdterr),
            ("lm_dof" => lm_dof)
        )

        return out_dict
    end

end
