# functions for fitting constrained lasso at fixed tuning parameter value

using Convex, SCS, ECOS
export lsq_constrsparsereg

"""
    lsq_constrsparsereg!(X, y, λ=0.0)

Sparse linear regression with constraints. Minimize
    `0.5sumabs2(√obswt .* (y - X * β) + λ * sumabs(penwt .* β)`
subject to linear constraints.

# Input
* `X`: predictor matrix.
* `y`: response vector.
* `λ`: tuning parameter. Default 0.

# Optional argument
* `A`: constraint matrix.
* `sense`: can be a vector of `'='`, `'<'`, or `'>'`.
* `b`: rhs of linear constaints.
* `lb`: lower bounds.
* `ub`: upper bounds.
* `obswt`: observation weights.
* `penwt`: predictor penalty weights. Default is `[0 1 1 ... 1]`
* `solver`: a solver Convex.jl can use.

# Output
* `β`: estimated coefficents.
* `objval`: optimal objective value.
* `problem`: Convex.jl problem.
"""
function lsq_constrsparsereg(
    X::AbstractMatrix,
    y::AbstractVector,
    λ::Number             = zero(eltype(X));
    A::AbstractMatrix     = zeros(eltype(X), 0, size(X, 2)),
    sense                 = '=',
    b                     = zeros(eltype(X), size(A, 1)),
    lb                    = typemin(eltype(X)),
    ub                    = typemax(eltype(X)),
    obswt::AbstractVector = ones(eltype(y), length(y)),
    penwt::AbstractVector = [zero(eltype(X)); ones(eltype(X), size(X, 2) - 1)],
    solver                = SCSSolver(verbose=0)
    )

    n, p = size(X)
    β = Variable(p)
    # objective
    problem = minimize((1//2)sumsquares(√obswt .* (y - X * β)) +
        λ * dot(penwt, abs(β)))
    # equality constraints
    if length(sense) == 1; sense = repmat([sense[1]], size(A, 1)); end
    if length(b) == 1; b = repmat([b[1]], size(A, 1)); end
    idx = find(sense .== '=')
    if length(idx) > 0
        problem.constraints += A[idx, :] * β == b[idx]
    end
    # inequality constraints
    idx = find(sense .== '<')
    if length(idx) > 0
        problem.constraints += A[idx, :] * β <= b[idx]
    end
    idx = find(sense .== '>')
    if length(idx) > 0
        problem.constraints += A[idx, :] * β >= b[idx]
    end
    # boundary conditions
    idx = find(lb .> typemin(eltype(X)))
    if length(idx) > 0
        problem.constraints += β[idx] >= lb[idx]
    end
    idx = find(ub .< typemax(eltype(X)))
    if length(idx) > 0
        problem.constraints += β[idx] <= ub[idx]
    end

    # optimization
    solve!(problem, solver)

    return β.value, problem.optval, problem

end
