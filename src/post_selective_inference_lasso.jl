"""
    Tests on model selection and Type 1 Error
"""
using Pkg
Pkg.status()

using GLM
using GLMNet
using RCall
using Distributions
using Random
using StatsPlots


"""
    1) p < n
"""
# Simulate data
n = 100
p = 20

# create true beta from random choice from [0.5, 1., -1., -0.5]
Random.seed!(124)
beta_true = Random.rand([0.5, 1., -1., -0.5], p-1)

# Set 5 coefficients to 0, to reflect that some features are to be excluded
beta_true[[2, 5, 10, 15, 19]] .= 0.

# Add coefficient for the intercept
pushfirst!(beta_true, 1.)


X = zeros(Float64, n, p)
# Intercept
X[:, 1] .= 1.

# Fill other columns with random Normal samples
norm_d = Distributions.Normal(0., 0.5)
X[:, 2:p] = Random.rand(norm_d, (n, p-1))
mean(X, dims=1)

# Get y = X * beta + err ~ N(0, 1)
y = X * beta_true + Random.rand(Distributions.Normal(), n)


# OLS
beta_hat_ols = X\y

scatter(beta_hat_ols, label="OLS")
xticks!([2, 5, 10, 15, 19] .+ 1)
scatter!(beta_true, label="True")
hline!([0], linestyle=:dash, color="black")

# Using the GLM library
lm1 = GLM.lm(X, y)
lm1_res = GLM.coeftable(lm1)
lm1_df = DataFrames.DataFrame(lm1_res)

scatter!(lm1_df[:, "Coef."], label="LM")

# Add column for significance
lm1_df[!, "Zero Coef"] = lm1_df[:, "Pr(>|t|)"] .> 0.05

withenv("LINES" => 1) do
    display(lm1_df[lm1_df[:, "Zero Coef"], :])
end

# So here we have correctly identified all the NULL variables

"""
    Use LASSO from GLMNet, then get the CI estimates using RCall
    and the package  SelectiveInference
"""
Xcovs = X[:, 2:p]
lasso_cv = GLMNet.glmnetcv(Xcovs, y, intercept=true)
argmin(lasso_cv.meanloss)
lambda_min = GLMNet.lambdamin(lasso_cv)
lasso_coeffs = GLMNet.coef(lasso_cv)
# lasso_cv.path

scatter!(lasso_coeffs, label="Lasso")


" Selective Inference "
# Use R library SelectiveInference to calculate CIs from Lasso regression
# @rput lasso_coeffs
# @rput lambda_min
@rput Xcovs
@rput y

# Run these lines in R terminal after $
library(selectiveInference)
library(glmnet)

lasso_cv = cv.glmnet(Xcovs, y)
lambda_min = lasso_cv$lambda.min
# Following the recommendation on the selective inference package
beta = coef(lasso_cv, s=lambda_min/dim(Xcovs)[1])[-1]
# beta = coef(lasso_cv)[-1]

fixedLassoInf(Xcovs, y, beta, lambda_min)

