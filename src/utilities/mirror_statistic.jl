module mirror_statistic

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
        r = sample(range(1, n), Int(n/2), replace=false)
        rc = setdiff(range(1, n), r)

        y1 = y[r]
        X1 = X[r, :]

        y2 = y[rc]
        X2 = X[rc, :]

        return (X1, X2, y1, y2)
    end

end
