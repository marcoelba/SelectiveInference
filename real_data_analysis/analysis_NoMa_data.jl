# Analysis of NoMa data
"""
    Outcome: Fasting Triglicerides
    Features: Gene expression data
"""

using CSV
using DataFrames
using Plots
using StatsBase
using LinearAlgebra
using MultivariateStats

script_path = normpath(joinpath(@__FILE__, "..", ".."))
include(joinpath(script_path, "src", "utilities", "randomisation_ds.jl"))

# Load data
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

# Some exploratory analysis
cor_mat = cor(X, dims=1)
heatmap(cor_mat)

# set diag to 0, not needed
cor_mat[LinearAlgebra.diagind(cor_mat)] .= 0.
(sum(abs.(cor_mat) .> 0.75) - size(X)[2]) / 2
cor_over_05 = abs.(cor_mat) .> 0.75

sum(sum(cor_over_05, dims=1) .!= 0)
sub_genes_names = genes_names[(sum(cor_over_05, dims=2) .!= 0)[:, 1]]
sub_cor_mat = cor_mat[
    (sum(cor_over_05, dims=2) .!= 0)[:, 1],
    (sum(cor_over_05, dims=1) .!= 0)[1,:]
]
heatmap(sub_cor_mat)

# Using Plotly for an interactive plot
using PlotlyJS
hm_plotly = PlotlyJS.plot(PlotlyJS.heatmap(
    x=sub_genes_names,
    y=sub_genes_names,
    z=sub_cor_mat
))

PlotlyJS.savefig(hm_plotly, "HM_sub_genes.pdf")
open("./HM_sub_genes.html", "w") do io
    PlotlyBase.to_html(io, hm_plotly.plot, autoplay=true)
end

# Check PCA
pca_model = MultivariateStats.fit(PCA, transpose(X), maxoutdim=size(X)[2]);
tvar(pca_model::PCA)
tprincipalvar(pca_model::PCA)

prop_explained_var = principalvars(pca_model) / tvar(pca_model)
cum_prop_explained_var = cumsum(prop_explained_var)

eigvals(pca_model)

plot(cum_prop_explained_var, marker=(:circle, 5), label="cumulative prop")
plot(prop_explained_var, marker=(:circle, 5), label="prop")
# no strong Principal components
# slow growth of variance explained


# Check KernelPCA
kpca_model = MultivariateStats.fit(KernelPCA, transpose(X), maxoutdim=size(X)[2]);

prop_explained_var = eigvals(kpca_model::KernelPCA) / sum(eigvals(kpca_model::KernelPCA))
cum_prop_explained_var = cumsum(prop_explained_var)

plot(cum_prop_explained_var, marker=(:circle, 5), label="cumulative prop")
plot(prop_explained_var, marker=(:circle, 5), label="prop")
# no strong Kernel Principal components
# slow growth of variance explained


"""
    Run Randomisation + Mirror Statistic
"""

# ---------------- FDR: 10% ----------------
res = randomisation_ds.real_data_rand_ms(
    y=y,
    X=X,
    gamma=1.,
    fdr_level=0.1,
    alpha_lasso=1.
)

println("Number of selected features: $(sum(res["selected_ms_coef"]))")
genes_names[res["selected_ms_coef"]]
# "ABCG1"
# The protein encoded by this gene is a member of the superfamily of ATP-binding 
# cassette (ABC) transporters.
# ABC proteins transport various molecules across extra- and intra-cellular 
# membranes. ABC genes are divided into seven distinct subfamilies 
# (ABC1, MDR/TAP, MRP, ALD, OABP, GCN20, White). 
# This protein is a member of the White subfamily.
# It is involved in macrophage cholesterol and phospholipids transport, 
# and may regulate cellular lipid homeostasis in other cell types. 
# Six alternative splice variants have been identified.

# "AI207942"


# LM coefs and p-values
res["lm_coef"][res["selected_ms_coef"]]
res["lm_pvalues"][res["selected_ms_coef"]]

# only pvalues
sum(res["lm_pvalues"] .<= 0.05)


# ---------------- FDR: 20% ----------------
res = randomisation_ds.real_data_rand_ms(
    y=y,
    X=X,
    gamma=1.,
    fdr_level=0.3,
    alpha_lasso=1.
)

println("Number of selected features: $(sum(res["selected_ms_coef"]))")
genes_names[res["selected_ms_coef"]]
# "ABCG1"
# "AI207942"
# "ALDH6A1"
# "ANKRD12"

# LM coefs and p-values
res["lm_coef"][res["selected_ms_coef"]]
res["lm_pvalues"][res["selected_ms_coef"]]

# only pvalues
sum(res["lm_pvalues"] .<= 0.05)
