# Outcome (Y) randomisation + Mirror Statistic (from DS)

module wrapper_pipeline_inference

    using GLM
    using GLMNet
    using Distributions
    using Random
    using StatsPlots
    using Plots
    using LinearAlgebra
    using DataFrames

    abs_project_path = normpath(joinpath(@__FILE__,".."))
    rel_path = joinpath(abs_project_path, "utilities")

    include(joinpath(rel_path, "data_generation.jl"))
    include(joinpath(rel_path, "randomisation_ds.jl"))
    include(joinpath(rel_path, "mirror_statistic.jl"))
    include(joinpath(rel_path, "classification_metrics.jl"))
    include(joinpath(rel_path, "variable_selection_plus_inference.jl"))


    """
        Run inference on data with the provided methods
    """
    function wrapper_inference(;
        data,
        estimate_sigma2=true,
        methods_to_evaluate=["Rand_MS", "DS", "MDS"],
        fdr_level=0.1,
        alpha_lasso=1.
        )

        metrics_array = []

        if "Rand_MS" in methods_to_evaluate
            rand_ms_selection = randomisation_ds.rand_ms(
                y=data.y,
                X=data.X,
                sigma2=data.sigma2,
                gamma=1.,
                estimate_sigma2=estimate_sigma2,
                fdr_level=fdr_level,
                alpha_lasso=alpha_lasso
            )

            push!(
                metrics_array,
                classification_metrics.wrapper_metrics(data.beta_true .!= 0, rand_ms_selection)
            )
        end

        if "DS" in methods_to_evaluate
            ds_selection = mirror_statistic.ds(X=data.X, y=data.y, fdr_level=fdr_level, alpha_lasso=alpha_lasso)

            push!(
                metrics_array,
                classification_metrics.wrapper_metrics(data.beta_true .!= 0, ds_selection)
            )
        end

        if "MDS" in methods_to_evaluate
            mds_selection = mirror_statistic.mds(
                X=data.X,
                y=data.y,
                n_ds=50,
                fdr_level=fdr_level,
                alpha_lasso=alpha_lasso
            )

            push!(
                metrics_array,
                classification_metrics.wrapper_metrics(data.beta_true .!= 0, mds_selection)
            )
        end

        return metrics_array

    end

end
