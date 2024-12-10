# Mirror Statistic assumptions violation
using CSV
using DataFrames
using Dates
using StatsPlots
using StatsBase
using RandMirror
using Random

abs_project_path = normpath(joinpath(@__FILE__,"..", ".."))

# Fixed parameters
n = 200
p = 1000
prop_non_zero_coef = 0.05
p1 = Int(p * prop_non_zero_coef)
p0 = p - p1

alpha_lasso = 1.
corr_factor = 0.5
fdr_level = 0.1

sigma2_plugin = 1.
label_files = "sigma_under_estimated"

n_simu = 50

fdr = []
tpr = []
n_coef_included = []
lasso_tpr = []
mirror_statistic_coefs = zeros(p, n_simu)
ols_coefs = zeros(p, n_simu)
lasso_coefs = zeros(p, n_simu)
n_null_above = []
n_null_below = []

Random.seed!(233)

for simu = 1:n_simu

    data = RandMirror.data_generation.linear_regression_data(
        n=n,
        p=p,
        sigma2=1.,
        beta_intercept=0.,
        covariance_matrix=zeros(1, 1),
        correlation_coefficients=[corr_factor],
        cov_like_MS_paper=true,
        block_covariance=true,
        beta_pool=[-1., 1.],
        beta_signal_strength=1.,
        prop_zero_coef=1 - prop_non_zero_coef
    )

    u, v, sigma2_est = RandMirror.randomisation_ds.randomisation(
        y=data.y,
        X=data.X,
        gamma=1.,
        estimate_sigma2=true,
        sigma2=sigma2_plugin
    )

    # LASSO
    lasso_coef = RandMirror.variable_selection_plus_inference.linear_lasso(
        X=data.X,
        y=u,
        add_intercept=true,
        alpha_lasso=alpha_lasso
    )
    push!(n_coef_included, sum(lasso_coef .!= 0))
    ltpr = RandMirror.classification_metrics.true_positive_rate(
        true_coef=data.beta_true .!= 0,
        estimated_coef=lasso_coef .!= 0
    )
    push!(lasso_tpr, ltpr)
    lasso_coefs[:, simu] = lasso_coef

    # Fix the LASSO subset
    # lasso_coef[p0+1:p] = data.beta_true[p0+1:p] .+ randn(p1)*0.1

    # OLS
    lm_coef = RandMirror.variable_selection_plus_inference.ols(
        X=data.X,
        y=v,
        add_intercept=true,
        lasso_coef=lasso_coef
    )
    ols_coefs[:, simu] = lm_coef

    " Add Mirror Statistic on top of randomisation "
    ms_coef = RandMirror.mirror_statistic.mirror_stat(lm_coef, lasso_coef)
    mirror_statistic_coefs[:, simu] = ms_coef
    scatter(ms_coef)
    # get FDR threshold
    optimal_t = RandMirror.mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

    metrics = RandMirror.classification_metrics.wrapper_metrics(
        data.beta_true .!= 0,
        ms_coef .> optimal_t
    )

    push!(fdr, metrics.fdr)
    push!(tpr, metrics.tpr)

    null_above = sum(ms_coef[1:p0] .> optimal_t)
    null_below = sum(ms_coef[1:p0] .< -optimal_t)

    push!(n_null_below, null_below)
    push!(n_null_above, null_above)

end


plt = scatter(lasso_tpr, fdr, label=false, xlabel="LASSO TPR", ylabel="FDR", labelfontsize=15, markersize=5)
hline!([0.1], color="red", linewidth=3, label="FDR target")
savefig(plt, joinpath(abs_project_path, "results", "lassotpr_vs_fdr_wrong.pdf"))

plt = histogram(fdr, label=false, xlabel="FDR", labelfontsize=15)
vline!([median(fdr)], color="red", linewidth=3, label="Median FDR")
savefig(plt, joinpath(abs_project_path, "results", "lassotpr_vs_fdr_right.pdf"))

# distribution of coefficients
scatter(lasso_coefs, label=false)
scatter(ols_coefs, label=false)
scatter(mirror_statistic_coefs, label=false)
scatter(mirror_statistic_coefs[:, 1])

hist_up = histogram(n_null_above, label=false, xlabel="# coefs", labelfontsize=15)
vline!([mean(n_null_above)], color="red", label="mean", linewidth=2)
title!("Null coefs > t")
hist_down = histogram(n_null_below, label=false, xlabel="# coefs", labelfontsize=15)
vline!([mean(n_null_below)], color="red", label="mean", linewidth=2)
title!("Null coefs < t")
plt = plot(hist_down, hist_up)
savefig(plt, joinpath(abs_project_path, "results", "NO_simmetry_histograms.pdf"))

plt = scatter(n_null_below, n_null_above, label=false, xlabel="Null coefs < t", ylabel="Null coefs > t", labelfontsize=15)
savefig(plt, joinpath(abs_project_path, "results", "NO_simmetry_scatter.pdf"))


# LASSO with and withOUT randomisation

n = 200
p = 1000
prop_non_zero_coef = 0.05
p1 = Int(p * prop_non_zero_coef)
p0 = p - p1

alpha_lasso = 1.
corr_factor = 0.5
fdr_level = 0.1

sigma2_plugin = 1.

n_simu = 50

rand_lasso_tpr = []
full_lasso_tpr = []
ds_lasso_tpr = []

Random.seed!(233)

for simu = 1:n_simu

    data = RandMirror.data_generation.linear_regression_data(
        n=n,
        p=p,
        sigma2=1.,
        beta_intercept=0.,
        covariance_matrix=zeros(1, 1),
        correlation_coefficients=[corr_factor],
        cov_like_MS_paper=true,
        block_covariance=true,
        beta_pool=[-1., 1.],
        beta_signal_strength=1.,
        prop_zero_coef=1 - prop_non_zero_coef
    )

    u, v, sigma2_est = RandMirror.randomisation_ds.randomisation(
        y=data.y,
        X=data.X,
        gamma=1.,
        estimate_sigma2=true,
        sigma2=sigma2_plugin
    )

    # LASSO
    rand_lasso_coef = RandMirror.variable_selection_plus_inference.linear_lasso(
        X=data.X,
        y=u,
        add_intercept=true,
        alpha_lasso=alpha_lasso
    )
    ltpr = RandMirror.classification_metrics.true_positive_rate(
        true_coef=data.beta_true .!= 0,
        estimated_coef=rand_lasso_coef .!= 0
    )
    push!(rand_lasso_tpr, ltpr)

    # FULL LASSO
    full_lasso_coef = RandMirror.variable_selection_plus_inference.linear_lasso(
        X=data.X,
        y=data.y,
        add_intercept=true,
        alpha_lasso=alpha_lasso
    )
    ltpr = RandMirror.classification_metrics.true_positive_rate(
        true_coef=data.beta_true .!= 0,
        estimated_coef=full_lasso_coef .!= 0
    )
    push!(full_lasso_tpr, ltpr)

    # DS LASSO
    ds = RandMirror.mirror_statistic.data_splitting(data.X, data.y)
    ds_lasso_coef = RandMirror.variable_selection_plus_inference.linear_lasso(
        X=ds.X1,
        y=ds.y1,
        add_intercept=true,
        alpha_lasso=alpha_lasso
    )
    ltpr = RandMirror.classification_metrics.true_positive_rate(
        true_coef=data.beta_true .!= 0,
        estimated_coef=ds_lasso_coef .!= 0
    )
    push!(ds_lasso_tpr, ltpr)
    
end

plt = violin([1], full_lasso_tpr, alpha=0.5, label=false)
violin!(rand_lasso_tpr, alpha=0.5, label=false)
violin!(ds_lasso_tpr, alpha=0.5, label=false)
xticks!([1, 2, 3], ["Full", "Rand", "DS"], tickfontsize=15)
savefig(plt, joinpath(abs_project_path, "results", "LASSO_comparison.pdf"))


# Distribution difference of M between DS and Rand
# Fixed parameters
n = 200
p = 1000
prop_non_zero_coef = 0.05
p1 = Int(p * prop_non_zero_coef)
p0 = p - p1

alpha_lasso = 1.
corr_factor = 0.5
fdr_level = 0.1

sigma2_plugin = 1.
label_files = "DS_Rand_M_distribution"

n_simu = 50

mirror_statistic_coefs_Rand = zeros(p, n_simu)
mirror_statistic_coefs_DS = zeros(p, n_simu)

Random.seed!(233)

for simu = 1:n_simu

    data = RandMirror.data_generation.linear_regression_data(
        n=n,
        p=p,
        sigma2=1.,
        beta_intercept=0.,
        covariance_matrix=zeros(1, 1),
        correlation_coefficients=[corr_factor],
        cov_like_MS_paper=true,
        block_covariance=true,
        beta_pool=[-1., 1.],
        beta_signal_strength=1.,
        prop_zero_coef=1 - prop_non_zero_coef
    )

    # RANDOMISATION
    u, v, sigma2_est = RandMirror.randomisation_ds.randomisation(
        y=data.y,
        X=data.X,
        gamma=1.,
        estimate_sigma2=true,
        sigma2=sigma2_plugin
    )

    # LASSO
    lasso_coef = RandMirror.variable_selection_plus_inference.linear_lasso(
        X=data.X,
        y=u,
        add_intercept=true,
        alpha_lasso=alpha_lasso
    )
    # Fix the LASSO subset
    # lasso_coef[p0+1:p] = data.beta_true[p0+1:p] .+ randn(p1)*0.1

    # OLS
    lm_coef = RandMirror.variable_selection_plus_inference.ols(
        X=data.X,
        y=v,
        add_intercept=true,
        lasso_coef=lasso_coef
    )

    # Add Mirror Statistic on top of randomisation
    ms_coef = RandMirror.mirror_statistic.mirror_stat(lm_coef, lasso_coef)
    mirror_statistic_coefs_Rand[:, simu] = ms_coef

    data_split = RandMirror.mirror_statistic.data_splitting(data.X, data.y)

    lasso_coef, lm_coef, lm_pvalues = RandMirror.variable_selection_plus_inference.lasso_plus_ols(
        X1=data_split.X1,
        X2=data_split.X2,
        y1=data_split.y1,
        y2=data_split.y2,
        add_intercept=true,
        alpha_lasso=alpha_lasso,
    )
    
    # Get Mirror Statistic
    ms_coef_ds = RandMirror.mirror_statistic.mirror_stat(lm_coef, lasso_coef)
    mirror_statistic_coefs_DS[:, simu] = ms_coef_ds

    # get FDR threshold
    # optimal_t = RandMirror.mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

    # metrics = RandMirror.classification_metrics.wrapper_metrics(
    #     data.beta_true .!= 0,
    #     ms_coef .> optimal_t
    # )

    # push!(fdr, metrics.fdr)
    # push!(tpr, metrics.tpr)

    # null_above = sum(ms_coef[1:p0] .> optimal_t)
    # null_below = sum(ms_coef[1:p0] .< -optimal_t)

    # push!(n_null_below, null_below)
    # push!(n_null_above, null_above)

    # Data Splitting

end


scatter(mirror_statistic_coefs_DS[:, 1])
scatter!(mirror_statistic_coefs_Rand[:, 1])

jj = 3
scatter(mirror_statistic_coefs_DS[jj, :])
scatter!(mirror_statistic_coefs_Rand[jj, :])

histogram(mirror_statistic_coefs_DS[p, :], alpha=0.5)
histogram!(mirror_statistic_coefs_Rand[p, :], alpha=0.5)

histogram(mirror_statistic_coefs_DS[1, :], alpha=0.5)
histogram!(mirror_statistic_coefs_Rand[1, :], alpha=0.5)

plt = scatter(lasso_tpr, fdr, label=false, xlabel="LASSO TPR", ylabel="FDR", labelfontsize=15, markersize=5)
hline!([0.1], color="red", linewidth=3, label="FDR target")
savefig(plt, joinpath(abs_project_path, "results", "lassotpr_vs_fdr_wrong.pdf"))

plt = histogram(fdr, label=false, xlabel="FDR", labelfontsize=15)
vline!([median(fdr)], color="red", linewidth=3, label="Median FDR")
savefig(plt, joinpath(abs_project_path, "results", "lassotpr_vs_fdr_right.pdf"))

# distribution of coefficients
scatter(lasso_coefs, label=false)
scatter(ols_coefs, label=false)
scatter(mirror_statistic_coefs, label=false)
scatter(mirror_statistic_coefs[:, 1])
