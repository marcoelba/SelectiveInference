# Outcome (Y) randomisation + Mirror Statistic (from DS)

using Pkg
Pkg.status()

using GLM
using GLMNet
using Distributions
using Random
using StatsPlots
using Plots

include("./utilities/data_generation.jl")
include("./utilities/randomisation_ds.jl")
include("./utilities/mirror_statistic.jl")
include("./utilities/classification_metrics.jl")


# Simulate data
n = 100
p = 10

Random.seed!(1345)
data = data_generation.linear_regression_data(
    n=n,
    p=p,
    prop_zero_coef=0.3,
    beta_intercept=1.,
    sigma2=1.
)

scatter(data.beta_true, label="TRUE")
hline!([0], label="Zero")

true_zero_coef = findall(data.beta_true .== 0)


# randomisation of Y
# U = Y + W
# W ~ N(0, gamma*Sigma), gamma > 0
sigma2 = 1.
u, v = randomisation_ds.randomisation(y=data.y, gamma=1., sigma2=sigma2)


" Perform variable selection on U using Lasso "
lasso_coef, lm_coef, lm_pvalues = randomisation_ds.lasso_plus_ols(
    X=data.X,
    u=u,
    v=v,
    add_intercept=true
)

scatter!(lm_coef, label="LM on V")

# Check FDR with just the coefficients from LM on randomised data
classification_metrics.false_discovery_rate(
    true_coef=data.beta_true,
    estimate_coef=lm_coef
)

# To properly check FDR we need to use the estimated pvalues from the LM regression
# and adjust them for the desired FDR level
adjusted_pvalues = classification_metrics.bh_correction(p_values=lm_pvalues, fdr_level=0.1)

classification_metrics.false_discovery_rate(
    true_coef=data.beta_true,
    estimate_coef=adjusted_pvalues[:, 3]
)


" Add Mirror Statistic on top of randomisation "
ms_coef = mirror_statistic.mirror_stat(lm_coef, lasso_coef)
# get FDR threshold
optimal_t = mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q = 0.1)

classification_metrics.false_discovery_rate(
    true_coef=data.beta_true,
    estimate_coef=ms_coef .> optimal_t
)
# 0.125
