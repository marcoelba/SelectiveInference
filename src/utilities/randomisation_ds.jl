# Randomisation technique for variable selection

module randomisation_ds
    using Distributions
    using Random
    using GLM
    using GLMNet

    """
        randomisation(;y, gamma, sigma2)
        
        Randomise the outcome Y with an added Gaussian noise of mean 0 and variance proportional to gamma * sigma2,
        where sigma2 is the variance of y

        Return (u = y + w, v = y - 1/gamma*w)
    """
    function randomisation(;y::Vector{Float64}, gamma::Float64, sigma2::Float64)
        n = length(y)
        w = Random.rand(Distributions.Normal(0, sqrt(gamma * sigma2)), n)
        u = y + w

        v = y - 1/gamma * w

        return (u=u, v=v)
    end

    """
        lasso_plus_ols(
            X::Matrix{Float64},
            u::Vector{Float64}, 
            v::Vector{Float64},
            add_intercept=true
            )

        Estimate LASSO on the randomised outcome U, then use OLS on the de-randomised outcome V
    """
    function lasso_plus_ols(;
        X::Matrix{Float64},
        u::Vector{Float64},
        v::Vector{Float64},
        add_intercept=true
    )
        # Lasso from GLMNet includes the intercept by default
        lasso_cv = GLMNet.glmnetcv(X, u)
        lasso_coef = GLMNet.coef(lasso_cv)
        # pushfirst!(lasso_coef, lasso_cv.path.a0[argmin(lasso_cv.meanloss)])

        # Non-0 coefficients
        non_zero = lasso_coef .!= 0

        # Now LM(OLS) estimate on V = Y - 1/gamma*W ~ N(mu, sigma2(1 + 1/gamma)I_n)
        # GLM in Matrix notation does NOT include the intercept by default
        X = X[:, non_zero]
        p = size(X)[2]
        if add_intercept
            X = hcat(X, ones(size(X)[1], 1))
            p += 1
        end
        lm_on_v = GLM.lm(X, v)

        if add_intercept
            lm_on_v_coef = GLM.coef(lm_on_v)[1:(p - 1)]
            lm_pvalues_subset = GLM.coeftable(lm_on_v).cols[4][1:(p - 1)]
            lm_coef_int = last(GLM.coef(lm_on_v))
        else
            lm_on_v_coef = GLM.coef(lm_on_v)
            lm_pvalues_subset = GLM.coeftable(lm_on_v).cols[4]
            lm_coef_int = 0.
        end

        lm_coef = zeros(length(lasso_coef))
        lm_pvalues = ones(length(lasso_coef)) # pvalue for excluded vars assumed to be 1
        lm_coef[non_zero] = lm_on_v_coef
        lm_pvalues[non_zero] = lm_pvalues_subset

        return (lasso_coef=lasso_coef, lm_coef=lm_coef, lm_pvalues=lm_pvalues)
    end

end
