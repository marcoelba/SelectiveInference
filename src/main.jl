# main script - run simulations

using CSV
using DataFrames
using Dates

include("./utilities/simulation_runner.jl")


# Fixed parameters
n = 800
p = 2000
prop_non_zero_coef = 0.025
n_replications = 2

# Variable quantities
corr_coefficients_vec = [0., 0.2, 0.4, 0.6, 0.8]
beta_signal_strength_vec = [3., 4., 5., 6., 7.]
n_combinations = length(corr_coefficients_vec) * length(beta_signal_strength_vec)

date_now = Dates.now()
current_date = string(Dates.year(date_now)) * string(Dates.month(date_now)) * string(Dates.day(date_now))

final_csv_file_name = "./simulation_n_$(n)_p_$(p)_time_$(current_date).csv"

println("Simulation started")

df_metrics_all = DataFrames.DataFrame()
combination_counter = 0
for corr_coeff in corr_coefficients_vec
    for beta_signal_strength in beta_signal_strength_vec
        global combination_counter += 1
        println("Running combination $combination_counter of $n_combinations")

        csv_file_name = "./simulation_n_$(n)_p_$(p)_rho_$(floor(Int, corr_coeff*10))_beta_$(floor(Int, beta_signal_strength))_time_$(current_date).csv"

        data_generation_params = (
            n=n,
            p=p,
            beta_intercept=1.,
            sigma2=1.,
            correlation_coefficients=[corr_coeff],
            cov_like_MS_paper=true,
            block_covariance=true,
            beta_signal_strength=beta_signal_strength,
            prop_non_zero_coef=prop_non_zero_coef
        )
        
        df_metrics = generate_predictions(
            n_replications=n_replications,
            data_generation_params=data_generation_params,
            fdr_level=0.1,
            estimate_sigma2=true,
            methods_to_evaluate=["Rand_MS", "DS", "MDS"]
        )
        
        CSV.write(csv_file_name, df_metrics)

        df_metrics[!, "rho"] .= corr_coeff
        df_metrics[!, "signal_strength"] .= beta_signal_strength

        # aggregate dataframes
        if combination_counter == 1
            df_metrics_all = df_metrics

        elseif combination_counter > 1
            append!(df_metrics_all, df_metrics)
        end
    end
end

println("Simulation finished")

CSV.write(final_csv_file_name, df_metrics_all)

println("All Files saved")
