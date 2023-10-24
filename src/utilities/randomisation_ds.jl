# Randomisation technique for variable selection

module randomisation_ds
    using Distributions
    using Random
    using GLM
    using GLMNet
    using LinearAlgebra

    abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))
    include(joinpath(abs_project_path, "utilities", "variable_selection_plus_inference.jl"))
    include(joinpath(abs_project_path, "utilities", "mirror_statistic.jl"))

    
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
                lasso_cv = GLMNet.glmnetcv(X, y)
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

        return (u=u, v=v)
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
        u, v = randomisation(
            y=y,
            X=X,
            gamma=gamma,
            estimate_sigma2=estimate_sigma2,
            sigma2=sigma2
        )

        " Perform variable selection on U using Lasso and Inference on V using OLS "
        lasso_coef, lm_coef, lm_pvalues = variable_selection_plus_inference.lasso_plus_ols(
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

end
