# Computational time

using Distributions
using Random
using StatsPlots
using Plots
using LinearAlgebra
using DataFrames
using BenchmarkTools

using RandMirror
using RandMirror: randomisation_ds, variable_selection_plus_inference, mirror_statistic

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))


# settings as used in the T-rex paper
p1 = 30.
n = 300
p = 20000
correlation_coefficients = [0.]
cov_like_MS_paper = true
block_covariance = true
beta_signal_strength = 5.
beta_pool = [-1., -0.5, 0.5, 1.]
prop_non_zero_coef = p1 / p

data = RandMirror.data_generation.linear_regression_data(
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

function mds_timing()
    mirror_statistic.mds(
        X=X,
        y=y,
        n_ds=50,
        fdr_level=0.1,
        alpha_lasso=1.
    )
end


# RAND MS
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30
@benchmark rand_ms_timing()

# MDS
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60
@benchmark mds_timing()


# times Rand
n_vars = [1000, 5000, 10000, 20000]
n_t = [0.54, 1.72, 3.8, 8.6]
n_mb = [85, 370, 730, 1400]

# times MDS
n_vars = [1000, 5000, 10000, 20000]
n_t_mds = [9.2, 18.6, 45, 72]
n_mb_mds = [1000, 4000, 7600, 15000]


# Plot
l = @layout [grid(1, 2)]
x_labels = ["1e3", "5e3", "1e4", "2e4"]

# Time
y_ticks = vcat(n_t[[1, 3]], n_t_mds)

p1 = plot(n_vars, n_t, label="RandMS")
plot!(n_vars, n_t_mds, label="MDS")
xlabel!("N. variables")
ylabel!("Time - seconds")
xticks!(n_vars, x_labels)
yticks!(y_ticks)

# memory
n_mb ./1024
y_ticks_m = round.(vcat([0.08, 0.5], n_mb_mds./1024), digits=2)

p2 = plot(n_vars, n_mb/1024, label="RandMS")
plot!(n_vars, n_mb_mds/1024, label="MDS")
xlabel!("N. variables")
ylabel!("Memory - GB")
xticks!(n_vars, x_labels)
yticks!(y_ticks_m)

all_p = plot(p1, p2, layout = l, thickness_scaling=1.)
plot(all_p)
savefig(all_p, joinpath(abs_project_path, "results", "computation_benchmark.pdf"))
