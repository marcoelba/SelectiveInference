# Classification metrics

module classification_metrics

    """
    false_discovery_rate(;
        true_coef::BitVector,
        estimated_coef::BitVector
        )

        Calculate the False Discovery Rate from a given set of coefficients
    """
    function false_discovery_rate(;
        true_coef::Union{Vector{Float64}, BitVector},
        estimated_coef::Union{Vector{Float64}, BitVector}
        )

        sum_coef = true_coef + estimated_coef
        TP = sum(sum_coef .== 2.)
        FP = sum((sum_coef .== 1.) .& (estimated_coef .== 1.))

        tot_predicted_positive = TP + FP

        if tot_predicted_positive > 0
            FDR = FP / tot_predicted_positive
        else
            FDR = 0.
            # println("Warning: 0 Positive predictions")
        end

        return FDR
    end

    """
        Calculate the True Positive Rate (aka Sensitivity, Recall, Hit Rate):
        TP / (TP + FN)

        true_positive_rate(;
            true_coef::Union{Vector{Float64}, BitVector},
            estimated_coef::Union{Vector{Float64}, BitVector}
        )
        # Arguments
        - `true_coef::BitVector`: boolean vector of true coefficients, '1' refers to a coef != 0 and '0' otherwise.
        - `estimated_coef::BitVector`: boolean vector of estimated coefficients, '1' refers to a coef != 0 and '0' otherwise.
    """
    function true_positive_rate(;
        true_coef::Union{Vector{Float64}, BitVector},
        estimated_coef::Union{Vector{Float64}, BitVector}
        )

        sum_coef = true_coef + estimated_coef
        TP = sum(sum_coef .== 2.)
        FN = sum((sum_coef .== 1.) .& (true_coef .== 1))

        TPR = TP / (TP + FN)

        return TPR
    end

    """
        power(;
            true_coef::Union{Vector{Float64}, BitVector},
            estimated_coef::Union{Vector{Float64}, BitVector}
            )
    """
    function power(;
        true_coef::Union{Vector{Float64}, BitVector},
        estimated_coef::Union{Vector{Float64}, BitVector}
        )

        sum_coef = true_coef + estimated_coef
        TP = sum(sum_coef .== 2.)
        actual_positives = sum(true_coef .== 1)

        Power = TP / actual_positives

        return Power
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

    """
    Compute collection of metrics
    """
    function wrapper_metrics(true_coef, pred_coef)
        # FDR
        fdr = false_discovery_rate(
            true_coef=true_coef,
            estimated_coef=pred_coef
        )

        # TPR
        tpr = true_positive_rate(
            true_coef=true_coef,
            estimated_coef=pred_coef
        )

        return (fdr=fdr, tpr=tpr)

    end

end
