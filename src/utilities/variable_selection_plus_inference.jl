# Variable Selection + Inference for splitted data

module variable_selection_plus_inference
    using GLM
    using GLMNet
    using Distributions
    using Random
    using LinearAlgebra


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

end
