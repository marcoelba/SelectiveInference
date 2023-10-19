# Simulation generation

using GLM
using GLMNet
using Distributions
using Random
using StatsPlots
using Plots
using DataFrames
using CSV
using LinearAlgebra

include("../wrapper_pipeline_inference.jl")


function generate_predictions(
    covariance_matrix,
    block_covariance_options,
    prop_non_zero_coef,
    n_replications,
    data_generation_params=(n, p, beta_intercept=1., sigma2=1.),
    gamma_randomisation=1.,
    fdr_level=0.1,
    estimate_sigma2=true
    )

    for cov_structure in block_covariance_options
        for corr_first in correlations_first_offdiag
            for corr_sec in correlations_second_offdiag
                for prop_non_zero in prop_non_zero_coef

                    Random.seed!(1345)
                    correlation_coefficients=[corr_first, corr_sec]
                    
                    # Initialise to 0
                    average_metrics = zeros(n_replications, length(keys(results.class_metrics)))

                    # Do an average over n replications for each combination
                    n_good_experiments = 0
                    for replica in range(1, n_replications)
                        try
                            results = wrapper_pipeline_inference.wrapper_randomisation_inference(
                                n=data_generation_params.n,
                                p=data_generation_params.p,
                                correlation_coefficients=data_generation_params.correlation_coefficients,
                                block_covariance=data_generation_params.cov_structure,
                                prop_zero_coef=1 - data_generation_params.prop_non_zero,
                                sigma2=data_generation_params.sigma2,
                                beta_intercept=data_generation_params.beta_intercept,
                                estimate_sigma2=estimate_sigma2,
                                gamma_randomisation=gamma_randomisation,
                                fdr_level=fdr_level
                            )

                            for (metric, value) in enumerate(results.class_metrics)
                                average_metrics[replica, metric] = value
                            end
                            n_good_experiments += 1
        
                        catch pipeline_error
                            println("Warning: ", pipeline_error)
                            continue
                        end
                        
                    end

                    push!(
                        df_metrics,
                        append!(
                            [
                                cov_structure,
                                corr_first,
                                corr_sec,
                                prop_non_zero
                            ],
                            average_metrics ./ n_good_experiments
                        )
                    )

                end
            end
        end
    end
end


p = 5
covariance_x = diagm(ones(p))
first_offdiag_corr_coeff = 0.5
p1 = p

diag_offset = 0
for ll in range(2, p1)
    diag_offset += 1
    cor_coef = first_offdiag_corr_coeff * (p1 - ll) / (p1 - 1)
    for kk in range(1, p1 - diag_offset)
        covariance_x[kk, kk + diag_offset] = cor_coef
        covariance_x[kk + diag_offset, kk] = cor_coef
    end
end
