# FDR control via DS and MDS

using Pkg
Pkg.status()

using GLM
using GLMNet
using Distributions
using Random
using StatsPlots
using Plots


# Simulate data
n = 100
p = 20

# create true beta from random choice from [0.5, 1., -1., -0.5]
Random.seed!(124)
beta_true = Random.rand([0.5, 1., -1., -0.5], p-1)
sum(beta_true)
# Add coefficient for the intercept
pushfirst!(beta_true, 1.)

# Set 5 coefficients to 0, to reflect that some features are to be excluded
beta_true[[2, 5, 10, 15, 19]] .= 0.

X = zeros(Float64, n, p)
# Intercept
X[:, 1] .= 1.

# Fill other columns with random Normal samples
norm_d = Distributions.Normal(0., 0.5)
X[:, 2:p] = Random.rand(norm_d, (n, p-1))
mean(X, dims=1)

# Get y = X * beta + err ~ N(0, 1)
y = X * beta_true + Random.rand(Distributions.Normal(), n)

scatter(beta_true, label="TRUE")
hline!([0])

# Estimate simple linear regression using GLM
lm_full = GLM.lm(X, y)
scatter!(GLM.coef(lm_full), label="Full LM", markersize=2)


"""
    Now using Data Splitting (DS) with Mirror statistic
"""

# take a random partition of the data
Random.seed!(2345)
r = sample(range(1, n), Int(n/2), replace=false)
rc = setdiff(range(1, n), r)

y1 = y[r]
X1 = X[r, :]

y2 = y[rc]
X2 = X[rc, :]

# Use Lasso on the first set and then OLS on the second set using only the features
# selected by Lasso

# GLMNet automatically includes the intercept
lasso_cv_2 = GLMNet.glmnetcv(X2[:, 2:p], y2)
lasso_2_coef = GLMNet.coef(lasso_cv_2)
pushfirst!(lasso_2_coef, lasso_cv_2.path.a0[argmin(lasso_cv_2.meanloss)])

# Non-0 coefficients
non_zero = lasso_2_coef .!= 0

lm_1 = GLM.lm(X1[:, non_zero], y1)
lm_1_coef = GLM.coef(lm_1)

ds_coeff = zeros(p)
ds_coeff[non_zero] = lm_1_coef

scatter!(ds_coeff, label="DS", markersize=3, color="red")

lm_ci_lower = zeros(p)
lm_ci_lower[non_zero] = GLM.confint(lm_1)[:, 1]
Plots.scatter!(lm_ci_lower, label="DS CI L",
    markersize=1.5, color="red"
)

lm_ci_upper = zeros(p)
lm_ci_upper[non_zero] = GLM.confint(lm_1)[:, 2]
scatter!(lm_ci_upper, label="DS CI U", markersize=1.5, color="red")


# Build the Mirror Statistic for DS
# Mj = sign(b1 * b2) * (|b1| + |b2|)
function mirror_stat(beta1, beta2)
    sign.(beta1 .* beta2) .* (abs.(beta1) .+ abs.(beta2))
end

mirror_coef = mirror_stat(ds_coeff, lasso_2_coef)

function optimal_tau(mirror_coef, fdr_q)

    optimal_t = 0
    t = 0
    for t in range(0, maximum(mirror_coef), length=100)
        n_left_tail = sum(mirror_coef .< -t)
        n_right_tail = sum(mirror_coef .> t)
        n_right_tail = ifelse(n_right_tail > 0, n_right_tail, 1)
    
        fdp = n_left_tail / n_right_tail

        if fdp <= fdr_q
            optimal_t = t
            break
        end
    end

    return optimal_t
end

optimal_t = optimal_tau(mirror_coef, 0.05)
mirror_coef .> optimal_t
ds_coeff[mirror_coef .> optimal_t]

findall(mirror_coef .> optimal_t)
