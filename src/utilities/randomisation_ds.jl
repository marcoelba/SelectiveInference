# Randomisation technique for variable selection

module randomisation_ds
    using Distributions
    using Random

    """
        randomisation(;y, gamma, sigma2)
        
        Randomise the outcome Y with an added Gaussian noise of mean 0 and variance proportional to gamma * sigma2,
        where sigma2 is the variance of y

        Return (u = y + w, v = y - 1/gamma*w)
    """
    function randomisation(;y::Vector{Float64}, gamma::Float64, sigma2::Float64)
        n = length(y)
        w = Random.rand(Distributions.Normal(0, sqrt(gamma * sigma2)), n)
        u = y + w

        v = y - 1/gamma * w

        return (u=u, v=v)
    end

end
