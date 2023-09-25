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
end
