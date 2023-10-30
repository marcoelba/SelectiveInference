# main script - run parallel simulations

using CSV
using DataFrames
using Dates

abs_project_path = normpath(joinpath(@__FILE__,"..", "..", "src"))
include(joinpath(abs_project_path, "utilities", "simulation_runner.jl"))

function single_csv_file_name(;base_experiment_dir, beta_value, rho_value)
    rho_value = floor(Int, rho_value*10)
    beta_value = floor(Int, beta_value)
    return joinpath(
        base_experiment_dir,
        "beta_$(beta_value)_rho_$(rho_value).csv"
        )
end

# Fixed parameters
n = 800
p = 2000
prop_non_zero_coef = 0.025
n_replications = 50
methods_to_evaluate=["Rand_MS", "DS", "MDS"]
alpha_enet = 0.7
use_beta_pool = true


# Variable quantities (choose between sign strenght and beta pool vec)
corr_coefficients_vec = [0., 0.2, 0.4, 0.5, 0.6, 0.8]

if use_beta_pool
    beta_signal_strength_vec = [1.]
    beta_pool = [-1., -0.8, -0.5, 0.5, 0.8, 1.]
    beta_type = "pool"
else
    beta_pool = []
    beta_signal_strength_vec = [3., 4., 5., 6., 7.]
    beta_type = "random"
end

n_combinations = length(corr_coefficients_vec) * length(beta_signal_strength_vec)

date_now = Dates.now()
current_date = string(Dates.year(date_now)) * string(Dates.month(date_now)) * string(Dates.day(date_now))

# Create directory for current simulation results output
dir_name = "n_$(n)_p_$(p)_alpha_enet_$(floor(Int, alpha_enet*10))_beta_type_$(beta_type)"
dir_path = joinpath(abs_project_path, "results", current_date, dir_name)
if isdir(dir_path) & (length(readdir(dir_path)) != 0)
    throw("Directory already exists! Exiting execution!")
end
mkpath(dir_path)

# file name for the global metrics csv file
global_csv_file_name = joinpath(dir_path, "all_metrics.csv")


println("Simulation started")

Threads.@threads for corr_coeff in corr_coefficients_vec
    Threads.@threads for beta_signal_strength in beta_signal_strength_vec

        csv_file_name = single_csv_file_name(base_experiment_dir=dir_path, beta_value=beta_signal_strength, rho_value=corr_coeff)

        data_generation_params = (
            n=n,
            p=p,
            beta_intercept=1.,
            sigma2=1.,
            correlation_coefficients=[corr_coeff],
            cov_like_MS_paper=true,
            block_covariance=true,
            beta_signal_strength=beta_signal_strength,
            beta_pool=beta_pool,
            prop_non_zero_coef=prop_non_zero_coef
        )
        
        df_metrics = generate_predictions(
            n_replications=n_replications,
            data_generation_params=data_generation_params,
            fdr_level=0.1,
            estimate_sigma2=true,
            methods_to_evaluate=methods_to_evaluate,
            alpha_lasso=alpha_enet
        )
        
        df_metrics[!, "rho"] .= corr_coeff
        df_metrics[!, "signal_strength"] .= beta_signal_strength

        # save csv
        CSV.write(csv_file_name, df_metrics)

    end
end

println("Parallel computation finished")

println("Merging results")

df_metrics_all = DataFrames.DataFrame()
all_files = readdir(dir_path)

for (ii, csv_file) in enumerate(all_files)
    df_metrics = CSV.read(joinpath(dir_path, csv_file), DataFrames.DataFrame)

    if ii == 1
        global df_metrics_all = df_metrics

    elseif ii > 1
        append!(df_metrics_all, df_metrics)
    end

    # remove the single file
    # rm(joinpath(dir_path, csv_file))
end

CSV.write(global_csv_file_name, df_metrics_all)

println("Merging done")
