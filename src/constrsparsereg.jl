"""
```
  lsq_constrsparsereg(X, y, ρ = zero(eltype(X));
      Aeq       :: AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
      beq       :: AbstractVector = zeros(eltype(X), size(Aeq, 1)),
      Aineq     :: AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
      bineq     :: AbstractVector = zeros(eltype(X), size(Aineq, 1)),
      obswt     :: AbstractVector = ones(eltype(y), length(y)),
      penwt     :: AbstractVector = ones(eltype(X), size(X, 2)),
      warmstart :: Bool = false,
      solver = SCSSolver(verbose=0, max_iters=10e8)
      )
```
Fit constrained lasso at fixed tuning parameter value(s) by minimizing
    `0.5sumabs2(√obswt .* (y - X * β)) + ρ * sumabs(penwt .* β)`
subject to linear constraints.

### Arguments
- `X`       : predictor matrix.
- `y`       : response vector.
- `ρ`       : tuning parameter. Can be a number or a list of numbers. Default 0.

### Optional arguments
- `Aeq`     : equality constraint matrix.
- `beq`     : equality constraint vector.
- `Aineq`   : inequality constraint matrix.
- `bineq`   : inequality constraint vector.
- `obswt`   : observation weights.
- `penwt`   : predictor penalty weights. Default is `[1 1 1 ... 1]`.
- `solver`  : a solver Convex.jl supports. Default is SCS. <http://convexjl.readthedocs.io/en/latest/solvers.html>
- `β0`      : starting point for warm start.

### Returns
- `β`       : estimated coefficents.
- `objval`  : optimal objective value.
- `problem` : Convex.jl problem.

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
    warmstart::Bool       = false,
    solver = SCSSolver(verbose=0, max_iters=10e8)
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

      if warmstart
          TT = STDOUT # save original STDOUT stream
          redirect_stdout()
          solve!(problem, solver; warmstart = i > 1? true : false)
          redirect_stdout(TT)
          push!(prob_vec, problem)
          optval_vec[i] = problem.optval
      else
          TT = STDOUT # save original STDOUT stream
          redirect_stdout()
          solve!(problem, solver)
          redirect_stdout(TT) # restore STDOUT
          push!(prob_vec, problem)
          optval_vec[i] = problem.optval
      end

      if length(ρ) == 1
        return β.value, problem.optval, problem
      end

      β̂[:, i] = β.value

    end # end of for loop

    return β̂, optval_vec, prob_vec

end # end of function
