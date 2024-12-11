# Simulation generation

using GLM
using GLMNet
using Distributions
using Random
using DataFrames
using LinearAlgebra

using RandMirror: wrapper_pipeline_inference, data_generation


function generate_single_prediction(;
    data_generation_params::NamedTuple,
    estimate_sigma2::Bool=true,
    methods_to_evaluate=["Rand_MS", "DS", "MDS"],
    fdr_level::Float64=0.1,
    alpha_lasso::Float64=1.
    )
    
    # generate data
    data = data_generation.linear_regression_data(
        n=data_generation_params.n,
        p=data_generation_params.p,
        beta_intercept=data_generation_params.beta_intercept,
        sigma2=data_generation_params.sigma2,
        correlation_coefficients=data_generation_params.correlation_coefficients,
        cov_like_MS_paper=data_generation_params.cov_like_MS_paper,
        block_covariance=data_generation_params.block_covariance,
        beta_signal_strength=data_generation_params.beta_signal_strength,
        beta_pool=data_generation_params.beta_pool,
        prop_zero_coef=1. - data_generation_params.prop_non_zero_coef,
        include_binary_covariates=data_generation_params.include_binary_covariates,
        prop_binary=data_generation_params.prop_binary    
    )
    # store the lowest eigenvalue (as a measure of how well-conditioned is the cov matrix)
    cov_eigval = LinearAlgebra.eigvals(data.covariance_matrix)[1]

    metrics_array = wrapper_pipeline_inference.wrapper_inference(
        data=data,
        estimate_sigma2=estimate_sigma2,
        methods_to_evaluate=methods_to_evaluate,
        fdr_level=fdr_level,
        alpha_lasso=alpha_lasso
    )

    return metrics_array, cov_eigval

end


function generate_predictions(;
    n_replications::Int64,
    data_generation_params::NamedTuple,
    fdr_level::Float64=0.1,
    estimate_sigma2::Bool=true,
    methods_to_evaluate::Array{String}=["Rand_MS", "DS", "MDS"],
    alpha_lasso::Float64=1.
    )

    # Initialise metrics to 0
    metrics_names = "FDR_" .* methods_to_evaluate
    append!(metrics_names, "TPR_" .* methods_to_evaluate)
    append!(metrics_names, ["successsfull_run"])
    append!(metrics_names, ["cov_eigval"])
    df_metrics = DataFrames.DataFrame([name => zeros(n_replications) for name in metrics_names])

    Random.seed!(1345)
    
    for replica in range(1, n_replications)
        try
            metrics_array, cov_eigval = generate_single_prediction(
                data_generation_params=data_generation_params,
                estimate_sigma2=estimate_sigma2,
                methods_to_evaluate=methods_to_evaluate,
                fdr_level=fdr_level,
                alpha_lasso=alpha_lasso
            )
        
            for (ii, method) in enumerate(methods_to_evaluate)
                df_metrics[replica, "TPR_" * method] = metrics_array[ii].tpr
                df_metrics[replica, "FDR_" * method] = metrics_array[ii].fdr
            end
            df_metrics[replica, "successsfull_run"] = 1.
            df_metrics[replica, "cov_eigval"] = cov_eigval

        catch pipeline_error
            println("Warning: ", pipeline_error)
            # if the run was a failure
            df_metrics[replica, "successsfull_run"] = 0.
            continue
        end
        
    end

    return df_metrics

end


# ---------------- TEST ---------------------
# p = 1000
# n = 800

# data_generation_params = (
#     n=n,
#     p=p,
#     beta_intercept = 1.,
#     sigma2 = 1.,
#     correlation_coefficients = [0.99],
#     cov_like_MS_paper=true,
#     block_covariance=true,
#     beta_signal_strength = 5.,
#     # beta_pool=[],
#     beta_pool=[-1.5, -1., -0.8, 0.8, 1., 1.5],
#     prop_non_zero_coef = 0.2,
#     include_binary_covariates=true,
#     prop_binary=0.5
# )

# data = data_generation.linear_regression_data(
#     n=data_generation_params.n,
#     p=data_generation_params.p,
#     beta_intercept=data_generation_params.beta_intercept,
#     sigma2=data_generation_params.sigma2,
#     correlation_coefficients=data_generation_params.correlation_coefficients,
#     cov_like_MS_paper=data_generation_params.cov_like_MS_paper,
#     block_covariance=data_generation_params.block_covariance,
#     beta_signal_strength=data_generation_params.beta_signal_strength,
#     beta_pool=data_generation_params.beta_pool,
#     prop_zero_coef=1 - data_generation_params.prop_non_zero_coef,
#     include_binary_covariates=true,
#     prop_binary=0.5
# )
# data.covariance_matrix
# LinearAlgebra.eigvals(data.covariance_matrix)
# data.beta_true

# # test inv
# p = 10
# cov_mat = LinearAlgebra.diagm(ones(p))
# diag_offset = 0
# cor_coefs = collect(LinRange(0.9, 0.6, p-1))
# for cor_coef in cor_coefs
#     diag_offset += 1
#     for kk in range(1, p - diag_offset)
#         cov_mat[kk, kk + diag_offset] = cor_coef
#         cov_mat[kk + diag_offset, kk] = cor_coef
#     end
# end

# LinearAlgebra.cholesky(cov_mat)
# LinearAlgebra.inv(cov_mat)
# LinearAlgebra.eigvals(cov_mat)

# # check DS
# # include(joinpath(abs_project_path, "utilities", "mirror_statistic.jl"))
# # include(joinpath(abs_project_path,  "utilities", "classification_metrics.jl"))

# # ds_selection = mirror_statistic.ds(X=data.X, y=data.y, fdr_level=fdr_level, alpha_lasso=alpha_lasso)
# # classification_metrics.wrapper_metrics(data.beta_true .!= 0, ds_selection)


# estimate_sigma2=true
# methods_to_evaluate=["Rand_MS", "DS"]
# fdr_level=0.1
# n_replications = 3
# alpha_lasso = 1.

# metrics = generate_single_prediction(
#     data_generation_params=data_generation_params,
#     estimate_sigma2=true,
#     methods_to_evaluate=methods_to_evaluate,
#     fdr_level=0.1,
#     alpha_lasso=alpha_lasso,
#     include_binary_covariates=true,
#     prop_binary=0.5
# )

# df_metrics = generate_predictions(
#     n_replications=n_replications,
#     data_generation_params=data_generation_params,
#     fdr_level=0.1,
#     estimate_sigma2=true,
#     methods_to_evaluate=methods_to_evaluate,
#     alpha_lasso=alpha_lasso
# )
