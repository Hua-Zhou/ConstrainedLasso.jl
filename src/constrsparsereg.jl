# functions for fitting constrained lasso at fixed tuning parameter value

"""
    lsq_constrsparsereg!(X, y, ρ=0.0)

Sparse linear regression with constraints. Minimize
    `0.5sumabs2(√obswt .* (y - X * β) + λ * sumabs(penwt .* β)`
subject to linear constraints.

# Input
* `X`: predictor matrix.
* `y`: response vector.
* `ρ`: tuning parameter. Can be a number of a list of numbers. Default 0.

# Optional argument
* `A`: constraint matrix.
* `sense`: can be a vector of `'='`, `'<'`, or `'>'`.
* `b`: rhs of linear constaints.
* `lb`: lower bounds.
* `ub`: upper bounds.
* `obswt`: observation weights.
* `penwt`: predictor penalty weights. Default is `[0 1 1 ... 1]`
* `solver`: a solver Convex.jl can use.
* `β0`: starting point for warm start.

# Output
* `β`: estimated coefficents.
* `objval`: optimal objective value.
* `problem`: Convex.jl problem.
"""

function lsq_constrsparsereg(
    X::AbstractMatrix,
    y::AbstractVector,
    ρ                     = zero(eltype(X));
    Aeq::AbstractMatrix   = zeros(eltype(X), 0, size(X, 2)),
    beq::AbstractVector   = zeros(eltype(X), size(Aeq, 1)),
    Aineq::AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    bineq::AbstractVector = zeros(eltype(X), size(Aineq, 1)),
    obswt::AbstractVector = ones(eltype(y), length(y)),
    penwt::AbstractVector = ones(eltype(X), size(X, 2)),
    #penwt::AbstractVector = [zero(eltype(X)); ones(eltype(X), size(X, 2) - 1)],
    solver                = SCSSolver(max_iters=10e6, verbose=0),
    warmstart::Bool       = false
    )

    n, p = size(X)

    β̂ = zeros(eltype(X), p, length(ρ))
    optval_vec = zeros(length(ρ))
    prob_vec = []



    for i in 1:length(ρ)

      β = Variable(p)
      ρi = ρ[i]
      if ρi == typemax(typeof(ρi))
          problem = minimize(dot(penwt, abs(β)))
      elseif ρi ≤ 0
          problem = minimize((1//2)sumsquares(sqrt(obswt) .* (y - X * β)))
      else
          problem = minimize((1//2)sumsquares(sqrt(obswt) .* (y - X * β)) +
              ρi * dot(penwt, abs(β)))
      end

      problem.constraints += Aeq * β == beq
      problem.constraints += Aineq * β <= bineq

      solve!(problem, solver)
      push!(prob_vec, problem)

      if warmstart
          solve!(problem, solver; warmstart = i > 1? true : false)
          optval_vec[i] = problem.optval
      else
          solve!(problem, solver)
          optval_vec[i] = problem.optval
      end

      if length(ρ) == 1
        return β.value, problem.optval, problem
      end

      β̂[:, i] = β.value

    end # end of for loop

    return β̂, optval_vec, prob_vec

end # end of function
