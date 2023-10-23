# main script - run simulations

using CSV
using DataFrames
using Dates

include("./utilities/data_generation.jl")
include("./utilities/simulation_runner.jl")


n = 800
p = 2000
corr_coeff=[0.]
prop_non_zero_coef = 0.025
beta_signal_strength = 7.

date_now = Dates.now()
current_date = string(Dates.year(date_now)) * string(Dates.month(date_now)) * string(Dates.day(date_now))
csv_file_name = "./simulation_n_$(n)_p_$(p)_time_$(current_date).csv"

data_generation_params = (
    n=n,
    p=p,
    beta_intercept=1.,
    sigma2=1.,
    correlation_coefficients=corr_coeff,
    cov_like_MS_paper=true,
    block_covariance=true,
    beta_signal_strength=beta_signal_strength,
    prop_non_zero_coef=prop_non_zero_coef
)

n_replications = 5

println("Simulation started")

df_metrics = generate_predictions(
    n_replications=n_replications,
    data_generation_params=data_generation_params,
    fdr_level=0.1,
    estimate_sigma2=true,
    methods_to_evaluate=["Rand_MS", "DS", "MDS"]
)
println("Simulation finished")

CSV.write(csv_file_name, df_metrics)
println("File saved to $csv_file_name")
