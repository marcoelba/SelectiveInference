# Analysis of NoMa data
"""
    Outcome: Fasting Triglicerides
    Features: Gene expression data
"""

using CSV
using DataFrames
using Plots
using StatsBase

script_path = normpath(joinpath(@__FILE__,".."))
include(joinpath(script_path, "src", "utilities", "randomisation_ds.jl"))


file_path_outcome = "/home/marco_ocbe/Documents/UiO_Postdoc/data/NoMa/NoMa_clin_outcome.csv"
file_path_features = "/home/marco_ocbe/Documents/UiO_Postdoc/data/NoMa/NoMa_microarray.csv"

df_y = CSV.read(file_path_outcome, DataFrame)
df_X = CSV.read(file_path_features, DataFrame)
size(df_X)
size(df_y)
genes_names = [nn for nn in names(df_X) if nn != "ID"]

Plots.histogram(df_y.fSTrig, label="Trigs")
sum(skipmissing(df_y.fSTrig) .== 0.)
minimum(skipmissing(df_y.fSTrig))

Plots.histogram(log.(y.fSTrig), label="Trigs")

# Merge y and X
df = DataFrames.innerjoin(df_y, df_X, on=:ID)
size(df)
# check for missing values
names(df)[mapcols(x -> any(ismissing.(x)), df)[1, :]]
mapcols(x -> any(ismissing.(x)), df) .== true

names(df)[collect(any(ismissing.(c)) for c in eachcol(df))]
collect(sum(ismissing.(c)) for c in eachcol(df))
DataFrames.dropmissing!(df, "fSTrig")

# Get arrays
y = df.fSTrig
X = Matrix(df[:, genes_names])

# Pre-processing
"""
    Take log of y
"""
function outcome_preprocessing(y, log_offset=1e-2)
    return log.(y .+ log_offset)
end

"""
    Sandardise numeric features to have mean 0 and sd 1 
"""
function features_preprocessing(X)
    col_mean = StatsBase.mean(X, dims=1)
    col_sd = StatsBase.std(X, dims=1)
    Xstd = (X .- col_mean) ./ col_sd
    return Xstd
end

y = outcome_preprocessing(y)
X = features_preprocessing(X)

Plots.histogram(y)
Plots.histogram(X[:, 1])
Plots.histogram(X[:, 2])


# Run Randomisation + Mirror Statistic
res = randomisation_ds.real_data_rand_ms(
    y=y,
    X=X,
    gamma=1.,
    fdr_level=0.1,
    alpha_lasso=1.
)

println("Number of selected features: $(sum(res["selected_ms_coef"]))")
genes_names[res["selected_ms_coef"]]

# LM coefs and p-values
res["lm_coef"][res["selected_ms_coef"]]
res["lm_pvalues"][res["selected_ms_coef"]]
