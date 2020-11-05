export bregman_div, kl_div, cross_entropy, differential_entropy

function bregman_div(x::Number, y::Number, f::Function)
    d_f = ForwardDiff.derivative(f,y)
    f(x) - f(y) - (x-y)*d_f
end

function bregman_div(x::Array, y::Array, f::Function)
    grad_f = ForwardDiff.gradient(f,y)
    f(x) - f(y) - (x-y)'*grad_f
end

# For details check "ENTROPIES AND CROSS-ENTROPIES OF EXPONENTIAL FAMILIES" by Nielsen and Nock
function kl_div(p::F, q::F) where F<:Distribution
    h_p, T_p, η_p, A_eval_p, A_p = exp_family(p)
    h_q, T_q, η_q, A_eval_q, A_q = exp_family(q)
    return bregman_div(η_q,η_p,A_p)
end

# https://en.wikipedia.org/wiki/Cross_entropy
# -int{q(x)logp(x)dx}
function cross_entropy(q::F, p::F) where F<:Distribution
    return differential_entropy(q) + kl_div(q,p)
end

function differential_entropy(p::Distribution)
    h_p, T_p, η_p, A_eval_p, A_p = exp_family(p)
    if length(η_p) > 1
        grad_A = ForwardDiff.gradient(A_p,η_p)
        return A_eval_p - η_p' * grad_A - baseE(p)
    else
        d_A = ForwardDiff.derivative(A_p,η_p)
        return A_eval_p - η_p * d_A - baseE(p)
    end
end
