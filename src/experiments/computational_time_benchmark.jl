# Computational time

using Distributions
using Random
using StatsPlots
using Plots
using LinearAlgebra
using DataFrames

using BenchmarkTools

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))
rel_path = joinpath(abs_project_path, "utilities")

include(joinpath(rel_path, "data_generation.jl"))
include(joinpath(rel_path, "randomisation_ds.jl"))
include(joinpath(rel_path, "variable_selection_plus_inference.jl"))

function fun_test(x)
    y = sqrt(x)
end
fun_test(10)

# timing
@btime fun_test(10)
@benchmark fun_test(10)


p1 = 10.
n = 300
p = 5000
correlation_coefficients = [0.5]
cov_like_MS_paper = true
block_covariance = true
beta_signal_strength = 5.
beta_pool = [-1., -0.5, 0.5, 1.]
prop_non_zero_coef = p1 / p

data = data_generation.linear_regression_data(
    n=n,
    p=p,
    beta_intercept=1.,
    sigma2=1.,
    correlation_coefficients=correlation_coefficients,
    cov_like_MS_paper=cov_like_MS_paper,
    block_covariance=block_covariance,
    beta_signal_strength=beta_signal_strength,
    beta_pool=beta_pool,
    prop_zero_coef=1. - prop_non_zero_coef
)
y = data.y
X = data.X

function rand_ms_timing()
    randomisation_ds.rand_ms(
        y=y,
        X=X,
        sigma2=1.,
        gamma=1.,
        estimate_sigma2=true,
        fdr_level=0.1,
        alpha_lasso=1.
    )
end

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20
@benchmark rand_ms_timing()

