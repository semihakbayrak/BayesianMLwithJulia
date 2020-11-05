export exp_family

#https://en.wikipedia.org/wiki/Exponential_family
# p(x) = h(x) exp{η'T(x) - A(η)}
function exp_family(p::Normal)
    h(x::Number) = 1/sqrt(2*pi)
    T(x::Number) = [x,x^2]
    η = [mean(p)/var(p), -0.5/var(p)]
    A_eval = 0.5*mean(p)^2/var(p) + log(std(p))
    A(η::Array) = - η[1]^2/(4*η[2]) - 0.5*log(-2*η[2])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func,  T_func, η, A_eval, A_func
end

function exp_family(p::MvNormal)
    k = length(mean(p))
    W = inv(cov(p))

    h(x::Number) = (2*pi)^(-k/2)
    T(x::Number) = [x;vec(x*x')]
    η = [W*mean(p);vec(-0.5*W)]
    A_eval = 0.5*(mean(p)'*W*mean(p) + logdet(cov(p)))
    A(η::Array) = - 0.25*η[1:k]'*inv(reshape(η[k+1:end],(k,k)))*η[1:k] -0.5*logdet(-2*reshape(η[k+1:end],(k,k)))

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func,  T_func, η, A_eval, A_func
end

function exp_family(p::Gamma)
    h(x::Number) = 1
    T(x::Number) = [log(x),x]
    η = [shape(p)-1, -rate(p)]
    A_eval = loggamma(shape(p)) - shape(p)*log(rate(p))
    A(η::Array) = loggamma(η[1]+1) - (η[1]+1)*log(-η[2])

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func,  T_func, η, A_eval, A_func
end

# We use variant 2 in https://en.wikipedia.org/wiki/Exponential_family
function exp_family(p::Dirichlet)
    h(x::Array) = 1
    T(x::Array) = log.(x)
    η = p.alpha .- 1
    A_eval = sum(loggamma.(p.alpha)) - loggamma(sum(p.alpha))
    A(η::Array) = sum(loggamma.(η .+ 1)) - loggamma(sum(η .+ 1))

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func,  T_func, η, A_eval, A_func
end

function exp_family(p::Wishart)
    V = p.S.mat
    ρ = size(V)[1]
    n = p.df

    h(x::Number) = 1
    T(x::Number) = [x;vec(logdet(x))]
    η = [vec(-0.5*inv(V)); (n-ρ-1)/2]
    A_eval = p.logc0
    A(η::Array) = -(η[end] + (ρ+1)/2)*log(det(-reshape(η[1:end-1],(ρ,ρ)))) +  logmvgamma(ρ,η[end]+(ρ+1)/2)

    h_func = (x)->h(x)
    T_func = (x)->T(x)
    A_func = (η)->A(η)

    return h_func,  T_func, η, A_eval, A_func
end
