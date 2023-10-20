# Simulation generation

using GLM
using GLMNet
using Distributions
using Random
using DataFrames
using LinearAlgebra

include("../wrapper_pipeline_inference.jl")
include("./data_generation.jl")
include("./randomisation_ds.jl")
include("./mirror_statistic.jl")
include("./classification_metrics.jl")
include("./variable_selection_plus_inference.jl")


function generate_single_prediction(;
    data_generation_params::NamedTuple,
    estimate_sigma2::Bool=true,
    methods_to_evaluate=["Rand_MS", "DS", "MDS"],
    fdr_level::Float64=0.1
    )
    
    # generate data
    data = data_generation.linear_regression_data(
        n=data_generation_params.n,
        p=data_generation_params.p,
        beta_intercept=data_generation_params.beta_intercept,
        sigma2=data_generation_params.sigma2,
        covariance_matrix=data_generation_params.covariance_matrix,
        beta_signal_strength=data_generation_params.beta_signal_strength,
        prop_zero_coef=data_generation_params.prop_non_zero_coef
    )

    metrics_array = wrapper_pipeline_inference.wrapper_inference(
        data=data,
        estimate_sigma2=estimate_sigma2,
        methods_to_evaluate=methods_to_evaluate,
        fdr_level=fdr_level
    )

    return metrics_array

end


function generate_predictions(;
    n_replications::Int64,
    data_generation_params::NamedTuple,
    fdr_level::Float64=0.1,
    estimate_sigma2::Bool=true,
    methods_to_evaluate::Array{String}=["Rand_MS", "DS", "MDS"]
    )

    # Initialise metrics to 0
    metrics_names = "FDR_" .* methods_to_evaluate
    append!(metrics_names, "TPR_" .* methods_to_evaluate)
    append!(metrics_names, ["successsfull_run"])
    df_metrics = DataFrames.DataFrame([name => zeros(n_replications) for name in metrics_names])

    Random.seed!(1345)
    
    n_good_experiments = 0
    for replica in range(1, n_replications)
        try
            metrics_array = generate_single_prediction(
                data_generation_params=data_generation_params,
                estimate_sigma2=estimate_sigma2,
                methods_to_evaluate=methods_to_evaluate,
                fdr_level=fdr_level
            )
        
            for (ii, method) in enumerate(methods_to_evaluate)
                df_metrics[replica, "TPR_" * method] = metrics_array[ii].tpr
                df_metrics[replica, "FDR_" * method] = metrics_array[ii].fdr
            end
            df_metrics[replica, "successsfull_run"] = 1.
            n_good_experiments += 1

        catch pipeline_error
            println("Warning: ", pipeline_error)
            # if the run was a failure
            df_metrics[replica, "successsfull_run"] = 0.
            continue
        end
        
    end

    return df_metrics

end


# TEST
covariance_matrix = diagm(ones(20))

data_generation_params = (
    n = 200,
    p = 20,
    beta_intercept = 1.,
    sigma2 = 1.,
    covariance_matrix = covariance_matrix,
    beta_signal_strength = 10.,
    prop_non_zero_coef = 0.5
)
estimate_sigma2=true
methods_to_evaluate=["Rand_MS", "DS", "MDS"]
fdr_level=0.1
n_replications = 5


results.data
results.class_metrics