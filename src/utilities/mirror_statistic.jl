module mirror_statistic

    using Random
    using Distributions

    include("./variable_selection_plus_inference.jl")


    """
        mirror_stat(beta1, beta2)

        Build the Mirror Statistic, given two coefficients estimate beta1 and beta2
        Mj = sign(b1 * b2) * (|b1| + |b2|)
    """
    function mirror_stat(beta1, beta2)
        sign.(beta1 .* beta2) .* (abs.(beta1) .+ abs.(beta2))
    end

    """
        optimal_threshold(mirror_coef, fdr_q)

        Calculate the optimal threshold that guarantees the give False Discovery Rate (fdr) at level q
    """
    function optimal_threshold(;mirror_coef, fdr_q)

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

    """
        data_splitting(X::AbstractArray, y::Vector{Float64})

        Produces a 50/50 data splitting of the given arrays
    """
    function data_splitting(X::AbstractArray, y::Vector{Float64})
        n = length(y)
        r = Distributions.sample(range(1, n), Int(n/2), replace=false)
        rc = setdiff(range(1, n), r)

        y1 = y[r]
        X1 = X[r, :]

        y2 = y[rc]
        X2 = X[rc, :]

        return (X1=X1, X2=X2, y1=y1, y2=y2)
    end

    """
    Single Data Splitting algorithm
    """
    function ds(;X::AbstractArray, y::Vector{Float64}, fdr_level::Float64)
        data_split = data_splitting(X, y)

        lasso_coef, lm_coef, lm_pvalues = variable_selection_plus_inference.lasso_plus_ols(
            X1=data_split.X1,
            X2=data_split.X2,
            y1=data_split.y1,
            y2=data_split.y2,
            add_intercept=true
        )
        
        # Get Mirror Statistic
        ms_coef = mirror_stat(lm_coef, lasso_coef)

        # get FDR threshold
        optimal_t = optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

        return ms_coef .> optimal_t

    end

    """
        mds(;X::AbstractArray, y::Vector{Float64}, n_ds::Int, fdr_level::Float64)

        Multiple Data Splitting
    """
    function mds(;X::AbstractArray, y::Vector{Float64}, n_ds::Int, fdr_level::Float64)
        p = size(X)[2]
        inclusion_matrix = zeros(p, n_ds)

        for m in range(1, n_ds)
            data_split = data_splitting(X, y)

            lasso_coef, lm_coef, lm_pvalues = variable_selection_plus_inference.lasso_plus_ols(
                X1=data_split.X1,
                X2=data_split.X2,
                y1=data_split.y1,
                y2=data_split.y2,
                add_intercept=true
            )
            
            # Get Mirror Statistic
            ms_coef = mirror_stat(lm_coef, lasso_coef)

            # get FDR threshold
            optimal_t = optimal_threshold(mirror_coef=ms_coef, fdr_q=fdr_level)

            # FDR MS
            included_coef = ms_coef .> optimal_t
            n_included = ifelse(sum(included_coef) > 0, sum(included_coef), 1)

            # Inclusion probability as defined in the MDS paper
            inclusion_matrix[:, m] = included_coef ./ n_included
        end

        inclusion_probs = sum(inclusion_matrix, dims=2)[:, 1] ./ n_ds
        
        # Find the selected coefficients

        # store the ranks in the original order
        sorted_probs = sort(inclusion_probs)
        cumsum_sorted_probs = cumsum(sorted_probs)
        probs_lower_fdr = cumsum_sorted_probs .<= fdr_level
        # get the largest prob that satisfy the above
        largest_prob = sorted_probs[sum(probs_lower_fdr)]
        
        return inclusion_probs .> largest_prob
        
    end
end
