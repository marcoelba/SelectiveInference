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


# settings as used in the T-rex paper
p1 = 30.
n = 300
p = 10000
correlation_coefficients = [0.]
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

Base.summarysize(X) / (1024^2)
Base.summarysize(y) / (1024^2)
Base.summarysize(data.covariance_matrix) / (1024^2)


# ------------- test memory size of big matrix -------------
k = 10000
# x_dist = Distributions.Normal{Float32}(0.f0, 1.f0)
x_dist = Distributions.Normal{Float64}(0., 1.)
X = rand(x_dist, k, k)
Base.summarysize(X) / (1024^2)

X = nothing
Base.GC.gc()


using GLMNet
using LARS

# LASSO
lasso_cv = GLMNet.glmnetcv(X, y, alpha=1., dfmax=30)
lasso_coef = GLMNet.coef(lasso_cv)
# Non-0 coefficients
non_zero = lasso_coef .!= 0
sum(non_zero)
# Time
@benchmark GLMNet.glmnetcv(X, y, alpha=1.)


# LARS
lars_est = LARS.lars(copy(X), y, method=:lasso, verbose=false)


# -----------------------------------------------------------------


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

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30
@benchmark rand_ms_timing()


# times
n_vars = [1000, 5000, 10000, 20000]
n_t = [0.54, 1.72, 3.8, 8.6]
n_mb = [85, 370, 730, 1400]

# Plot
l = @layout [grid(1, 2)]
x_labels = ["1e3", "5e3", "1e4", "2e4"]

# Time
p1 = plot(n_vars, n_t, label="time - seconds")
xlabel!("N. variables")
ylabel!("Seconds")
xticks!(n_vars, x_labels)
yticks!(n_t)

# memory
p2 = plot(n_vars, n_mb/1024, label="mem - GB")
xlabel!("N. variables")
ylabel!("GB")
xticks!(n_vars, x_labels)
yticks!(round.(n_mb/1024, digits=2))

all_p = plot(p1, p2, layout = l, thickness_scaling=1.)
plot(all_p)
savefig(all_p, joinpath(abs_project_path, "results", "computation_benchmark.pdf"))
