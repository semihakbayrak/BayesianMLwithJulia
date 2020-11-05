export logmean, logdetmean
#Some expectation quantities that does not exist in Distributions.jl


# Expectations related to univariate Normal distribution
# Expectation of log base measure h(x)
baseE(p::Normal) = log(1/sqrt(2*pi))

# Expectations related to multivariate Normal distribution
# Expectation of log base measure h(x)
baseE(p::MvNormal) = log((2*pi)^(-length(mean(p))/2))


# Expectations related to Gamma distribution
# Expectation of log base measure h(x)
baseE(p::Gamma) = 0
# E[log(x)]
logmean(p::Gamma) = digamma(shape(p)) - log(rate(p))

# Expectations related to Dirichlet distribution
# Expectation of log base measure h(x), variant 2
baseE(p::Dirichlet) = 0
# E[log(x)]
logmean(p::Dirichlet) = digamma.(p.alpha) .- digamma(sum(p.alpha))

# Expectations related to Wishart distribution
# Expectation of log base measure h(x)
baseE(p::Wishart) = 0
#multivariate digamma function
function mvdigamma(ρ, α)
    lmg(α) = logmvgamma(ρ,α)
    return ForwardDiff.derivative(lmg,α)
end
# E[log|x|]
function logdetmean(p::Wishart)
    V = p.S.mat
    n = p.df
    ρ = size(V)[1]
    return mvdigamma(ρ, n/2) + ρ*log(2) +logdet(V)
end
