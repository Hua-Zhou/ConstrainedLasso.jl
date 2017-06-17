# functions for fitting constrained lasso at fixed tuning parameter value

using MathProgBase, Ipopt

"""
    lsq_constrsparsereg!(β, X, y, λ=0.0)

Sparse linear regression with constraints. Minimize `0.5sumabs2(obswt .* (y - X * β) + sumabs()`
subject to linear constraints.

# Input
* `X`:
"""
function lsq_constrsparsereg!(
    β::AbstractVector,
    X::AbstractMatrix,
    y::AbstractVector,
    λ::Number = 0.0;
    Aeq::AbstractMatrix   = zeros(eltype(X), 0, size(X, 2)),
    beq::AbstractVector   = zeros(eltype(X), size(Aineq, 1)),
    Aineq::AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    bineq::AbstractVector = zeros(eltype(X), size(Aineq, 1)),
    obswt::AbstractVector = ones(eltype(y), length(y)),
    penwt::AbstractVector = [zero(eltype(X)); ones(eltype(X), size(X, 2) - 1)
    )



end
