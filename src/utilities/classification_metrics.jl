# Classification metrics

module classification_metrics

    """
        false_discovery_rate(;true_coef, estimate_coef)

        Calculate the False Discovery Rate from a given set of coefficients
    """
    function false_discovery_rate(;
        true_coef::Union{Vector{Float64}, BitVector},
        estimate_coef::Union{Vector{Float64}, BitVector})

        true_coef_bin = 1 .- (true_coef .== 0)
        estimate_coef_bin = 1 .- (estimate_coef .== 0)

        sum_coef = true_coef_bin + estimate_coef_bin
        TP = sum(sum_coef .== 2.)
        TN = sum(sum_coef .== 0.)
        FP = sum((sum_coef .== 1.) .& (estimate_coef_bin .== 1.))

        FDR = FP / (TP + FP)

        return FDR
    end


    """
        bh_correction(;p_values::Vector{Float64}, fdr_level::Float64)
        
        False Discovery Rate adjusted p-values using Benjamini-Hochberg procedure
    """
    function bh_correction(;p_values::Vector{Float64}, fdr_level::Float64)
        if fdr_level > 1.
            throw(error("fdr_level MUST be <= 1"))
        end

        # store the ranks in the original order
        pvalues_ranks_original = hcat(p_values, invperm(sortperm(p_values)))

        n_tests = length(p_values)
        sorted_p_values = hcat(sort(p_values), sortperm(sort(p_values)))
        adjusted_pvalues = (sorted_p_values[:, 2] ./ n_tests) .* fdr_level
        sorted_p_values = hcat(sorted_p_values, adjusted_pvalues)

        # find the largest p-value that is lower than its corresponding BH critical value
        ranks_significant_pvalues = sorted_p_values[sorted_p_values[:, 1] .< sorted_p_values[:, 3], 2]
        pvalues_ranks_original = hcat(pvalues_ranks_original, zeros(n_tests))
        rank = 1
        for rank in range(1, n_tests)
            pvalues_ranks_original[rank, 3] = ifelse(pvalues_ranks_original[rank, 2] in ranks_significant_pvalues, 1, 0)
        end

        return pvalues_ranks_original
    end

end
