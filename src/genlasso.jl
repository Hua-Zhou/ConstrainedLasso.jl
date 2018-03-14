"""
```
  genlasso(
      X    :: AbstractMatrix{T},
      y    :: AbstractVector{T};
      path :: Bool = true,
      ρ    :: Union{AbstractVector, Number} = zero(T),
      D    :: AbstractMatrix{T} = eye(size(X, 2)),
      solver = ECOSSolver(maxit=10e8, verbose=0)
    )
```
Solve generalized lasso problem by reformulating it as constrained lasso problem.
Note genralized lasso minimizes
    `0.5sumabs2(√obswt .* (y - X * β)) + ρ * sumabs(D * β)`.

### Arguments
- `X`       : predictor matrix.
- `y`       : response vector.

### Optional arguments
- `path`    : set `path=false` if user wishes to supply parameter value(s).
              Default is true.
- `ρ`       : tuning parameter value(s). Default is 0.
- `D`       : penalty matrix. Default is identity matrix.
- `solver`  : a solver Convex.jl supports. Default is ECOS.
              Note that Mosek and Gurobi are more robust than ECOS. Unlike ECOS or
              SCS, both Mosek and Gurobi require a license (free for academic
              use). <http://convexjl.readthedocs.io/en/latest/solvers.html>

### Returns
- `β`       : estimated coefficents.
- `objval`  : optimal objective value.
- `problem` : Convex.jl problem.
"""

function genlasso(
    X::AbstractMatrix{T},
    y::AbstractArray{T};
    path::Bool = true,
    ρ::Union{AbstractVector, Number} = zero(T),
    D::AbstractMatrix{T} = eye(size(X, 2)),
    solver = ECOSSolver(maxit=10e8, verbose=0)
    ) where T

  # singular value decomposition on D
  m = size(D, 1)
  F = svdfact!(D, thin = false)
  # extract singular values and submatrices
  singvals = F[:S]
  rankD = countnz(F[:S] .> abs(F[:S][1]) * eps(F[:S][1]) * maximum(size(D)))
  V1 = F[:V][:, 1:rankD]
  V2 = F[:V][:, rankD+1:end]
  U1 = F[:U][:, 1:rankD]
  U2 = F[:U][:, rankD+1:end]
  # calculate the MP-inverse of D
  Dplus = V1 * broadcast(*, U1', 1./F[:S])
  # transform the design matrix
  XDplus = X * Dplus
  XV2 = X * V2
  # projection matrix onto C(XV2)
  Pxv2 = (1 / dot(XV2, XV2)) * A_mul_Bt(XV2, XV2)
  # orthogonal projection matrix
  Mxv2 = eye(size(XV2, 1)) - Pxv2
  # obtain the new design matrix and response vector
  ỹ = vec(Mxv2 * y)
  X̃ = Mxv2 * XDplus
  # solve the constrained lasso problem
  if path
    α̂path, ρpath, = lsq_classopath(X̃, ỹ; solver=solver, Aeq = U2',
          beq = zeros(size(U2, 2)))
  else
    α̂path, = lsq_constrsparsereg(X̃, ỹ, ρ; solver=solver, Aeq = U2',
          beq = zeros(size(U, 2)))
    ρpath = ρ
  end
  # transform back to beta
  β̂path = Base.LinAlg.BLAS.ger!(1.0, vec(V2 * ((1 / dot(XV2, XV2)) *
    		At_mul_B(XV2, y))), ones(size(ρpath)), (eye(size(V2, 1)) -
    		V2 * ((1 / dot(XV2, XV2)) * At_mul_B(XV2, X))) * Dplus * α̂path)

  return β̂path, ρpath

end # end of function
