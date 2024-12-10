# variance wrongly estimated
using CSV
using DataFrames
using Dates
using StatsPlots
using StatsBase
using RandMirror
using LaTeXStrings

abs_project_path = normpath(joinpath(@__FILE__,"..", ".."))

# Fixed parameters
n = 100
p = 200
prop_non_zero_coef = 0.1
p * prop_non_zero_coef

alpha_lasso = 1.
corr_factor = 0.2
fdr_level = 0.1

sigma2_plugin = 0.1
label_files = "sigma_under_estimated"

n_simu = 50

fdr = []
tpr = []
n_coef_included = []

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
        estimate_sigma2=false,
        sigma2=sigma2_plugin
    )

    " Perform variable selection on U using Lasso and Inference on V using OLS "
    lasso_coef, lm_coef, lm_pvalues, lm_coef_int, lm_sdterr, lm_dof = RandMirror.variable_selection_plus_inference.lasso_plus_ols(
        X1=data.X,
        X2=data.X,
        y1=u,
        y2=v,
        add_intercept=true,
        alpha_lasso=alpha_lasso
    )
    push!(n_coef_included, sum(lasso_coef .!= 0))

    " Add Mirror Statistic on top of randomisation "
    ms_coef = RandMirror.mirror_statistic.mirror_stat(lm_coef, lasso_coef)
    # get FDR threshold
    optimal_t = RandMirror.mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

    metrics = RandMirror.classification_metrics.wrapper_metrics(
        data.beta_true .!= 0,
        ms_coef .> optimal_t
    )

    push!(fdr, metrics.fdr)
    push!(tpr, metrics.tpr)

end

plt_box = boxplot([1], fdr, label=false, color="blue", fillalpha=0.7, linewidth=2)
boxplot!([2], tpr, label=false, color="blue", fillalpha=0.7, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

plt_hist = histogram(n_coef_included, label=false, xlabel="n. variables included", labelfontsize=15)
vline!([mean(n_coef_included)], linewidth=3, label="Mean")

plt = plot(plt_box, plt_hist)
savefig(plt, joinpath(abs_project_path, "results", "$(label_files)_fdrtpr_boxplot.pdf"))


# sigma vs fdr

fdr = []
tpr = []
n_coef_included = []

sigma_range = range(0.1, 5., length=20)

n_simu = 20

for sigma2_plugin in sigma_range

    fdr_partial = []
    tpr_partial = []
    n_coef_included_partial = []

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
            estimate_sigma2=false,
            sigma2=sigma2_plugin
        )

        " Perform variable selection on U using Lasso and Inference on V using OLS "
        lasso_coef, lm_coef, lm_pvalues, lm_coef_int, lm_sdterr, lm_dof = RandMirror.variable_selection_plus_inference.lasso_plus_ols(
            X1=data.X,
            X2=data.X,
            y1=u,
            y2=v,
            add_intercept=true,
            alpha_lasso=alpha_lasso
        )

        " Add Mirror Statistic on top of randomisation "
        ms_coef = RandMirror.mirror_statistic.mirror_stat(lm_coef, lasso_coef)
        # get FDR threshold
        optimal_t = RandMirror.mirror_statistic.optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

        metrics = RandMirror.classification_metrics.wrapper_metrics(
            data.beta_true .!= 0,
            ms_coef .> optimal_t
        )

        push!(fdr_partial, metrics.fdr)
        push!(tpr_partial, metrics.tpr)
        push!(n_coef_included_partial, sum(lasso_coef .!= 0))

    end

    push!(fdr, mean(fdr_partial))
    push!(tpr, mean(tpr_partial))
    push!(n_coef_included, mean(n_coef_included_partial))

end

plt = scatter(sigma_range, fdr, label="Average FDR", xlabel=L"$\sigma^2$", labelfontsize=15, markersize=3, color="red")
scatter!(sigma_range, tpr, label="Average Power", markersize=3, color="blue")
scatter!(sigma_range, n_coef_included ./ p, label="Prop. Vars In", markersize=3, color="green")
vline!([1], linewidth=2, label=false, linestyle=:dash, color="darkgrey")
hline!([0.1], linewidth=2, label=false, linestyle=:dash, color="darkgrey")

savefig(plt, joinpath(abs_project_path, "results", "sigma_vs_fdr.pdf"))
