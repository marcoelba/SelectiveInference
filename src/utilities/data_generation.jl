# Module for data generation

module data_generation

    using Distributions
    using Random
    using LinearAlgebra


    """
    gen_linear_regression_data(;
        n::Int64,
        p::Int64,
        beta_intercept::Float64=0.,
        sigma2::Float64,
        correlated_covariates::bool=false,
        beta_pool::Vector{Float64}=[-1., -0.8, -0.5, 0.5, 0.8, 1.],
        prop_zero_coef::Float64=0.,
        dtype=Float64
    )

    Generate data for a linear regression of n observation and p covariates
    """
    function linear_regression_data(;
        n::Int64,
        p::Int64,
        beta_intercept::Float64=0.,
        sigma2::Float64,
        correlation_coefficients::Union{Vector{Float64}, Vector{Any}}=[],
        block_covariance::Bool=false,
        beta_pool::Vector{Float64}=[-1., -0.8, -0.5, 0.5, 0.8, 1.],
        prop_zero_coef::Float64=0.,
        dtype=Float64
    )
        if prop_zero_coef >= 1.
            throw(error("prop_zero_coef MUST be < 1"))
        end
        if sigma2 <= 0.
            throw(error("The Normal error varaince sigma2 MUST be > 0"))
        end

        beta_true = Random.rand(beta_pool, p)

        # Set coefficients to 0
        if prop_zero_coef > 0.
            n_zero_coef = floor(Int, p * prop_zero_coef)
            # If the X cov structure is block diagonal, set the first n0 coeff to 0, otherwise random selection
            if block_covariance
                which_zero = range(1, n_zero_coef)
            else
                which_zero = sample(range(1, p), n_zero_coef, replace=false)
            end
            beta_true[which_zero] .= 0.
        end

        # Generate X from a multivariate Normal distribution
        if block_covariance
            covariance_x = create_block_diagonal_toeplitz_matrix(
                p=p,
                p_blocks=[n_zero_coef, p-n_zero_coef],
                correlation_coefficients=correlation_coefficients
            )
        else
            covariance_x = create_toeplitz_covariance_matrix(p=p, correlation_coefficients=correlation_coefficients)
        end
        x_distr = Distributions.MultivariateNormal(covariance_x)
        X = transpose(Random.rand(x_distr, n))

        # Get y = X * beta + err ~ N(0, 1)
        y = X * beta_true + Random.rand(Distributions.Normal(0., sqrt(sigma2)), n)
        if beta_intercept > 0
            y .+= beta_intercept
        end

        return (y=y, X=X, beta_true=beta_true)

    end


    """
        create_toeplitz_covariance_matrix(;p, correlation_coefficients::Union{Vector{Float64}, Vector{Any}}=[])

        Generate a covariance matrix with a Toepliz structure, given the provided correlation coefficient for the main off-diagonal entries.
        Default is the (diagonal) Identity matrix.
    """
    function create_toeplitz_covariance_matrix(;p, first_offdiag_corr_coeff::Union{Vector{Float64}, Vector{Any}}=[])
        covariance_x = diagm(ones(p))
        if length(first_offdiag_corr_coeff) > 0
            diag_offset = 0
            for cor_coef in first_offdiag_corr_coeff
                diag_offset += 1
                for kk in range(1, p - diag_offset)
                    covariance_x[kk, kk + diag_offset] = cor_coef
                    covariance_x[kk + diag_offset, kk] = cor_coef
                end
            end
        end

        return covariance_x

    end

    """
        create_block_diagonal_toeplitz_matrix(;p, correlation_coefficients::Union{Vector{Float64}, Vector{Any}}=[])

        Create a block diagonal matrix with 2 blocks
    """
    function create_block_diagonal_toeplitz_matrix(;p, p_blocks, correlation_coefficients::Union{Vector{Float64}, Vector{Any}}=[])
        covariance_full = diagm(ones(p))
        # make 2 blocks
        
        start_position = 1
        end_position = 0
        for p_block in p_blocks
            covariance_block = create_toeplitz_covariance_matrix(p=p_block, correlation_coefficients=correlation_coefficients)
            end_position += p_block
            covariance_full[start_position:end_position, start_position:end_position] = covariance_block
            start_position += p_block
        end
        
        return covariance_full
    end

end
