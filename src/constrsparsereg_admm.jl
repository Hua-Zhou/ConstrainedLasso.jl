"""
```
  lsq_constrsparsereg_admm(X, y, ρ::Number = zero(eltype(X));
    proj         = x -> x,
    obswt        :: AbstractVector = ones(eltype(y), length(y)),
    penwt        :: AbstractVector = ones(eltype(X), size(X, 2)),
    β0           :: AbstractVector = zeros(eltype(X), size(X, 2)),
    admmmaxite   :: Number = 10000,
    admmabstol   :: Number = 1e-4,
    admmreltol   :: Number = 1e-4,
    admmscale    :: Number = 1 / length(y),
    admmvaryscale:: Bool = false
    )
```
Fit constrained lasso at fixed tuning parameter value by applying the
alternating direction method of multipliers (ADMM) algorithm.

### Arguments
- `X`       : predictor matrix.
- `y`       : response vector.
- `ρ`       : tuning parameter. Default 0.

### Optional arguments
- `proj`         : projection onto the constraint set. Default is identity.
- `obswt`        : observation weights.
- `penwt`        : predictor penalty weights. Default is `[1 1 1 ... 1]`.
- `β0`           : starting point.
- `admmmaxite`   : maximum number of iterations for ADMM. Default is `10000`.
- `admmabstol`   : absolute tolerance for ADMM.
- `admmreltol`   : relative tolerance for ADMM.
- `admmscale`    : ADMM scale parameter. Default is `1/n`.
- `admmvaryscale`: dynamically chance the ADMM scale parameter. Default is false.

### Returns
- `β`       : estimated coefficents.
"""

function lsq_constrsparsereg_admm(
    X::AbstractMatrix{T},
    y::AbstractVector{N},
    ρ::Number        = zero(eltype(X));
    proj::Function   = x -> x,
    obswt::Vector{N} = ones(eltype(y), length(y)),
    penwt::Vector{T} = ones(eltype(X), size(X, 2)),
    β0::Vector{T}    = zeros(eltype(X), size(X, 2)),
    admmmaxite::Int       = 10000,
    admmabstol::Float64   = 1e-4,
    admmreltol::Float64   = 1e-4,
    admmscale::Float64    = 1 / length(y),
    admmvaryscale::Bool   = false
    ) where T where N

    n, p = size(X)
    β = copy(β0)
    z = proj(β)
    u = β - z
    v = similar(β)
    zold = similar(z)

    # allocate working arrays
    Xaug = vcat(X, eye(p))
    yaug = vcat(y, fill(1, p))
    obswtaug = vcat(obswt, fill(1, p))
    λ = [ρ / (n + p)]

    for iter in 1:admmmaxite

        # update working arrays
        for j in 1:p
          admmscaleinv = 1 / √admmscale
          if iter == 1 || admmvaryscale
            Xaug[n+j, j] = admmscaleinv
          end
          yaug[n+j] = (z[j] - u[j]) * admmscaleinv
        end
        # update β - lasso
        path = glmnet(Xaug, yaug;
                weights = obswtaug, lambda = λ, penalty_factor = penwt,
                standardize = false, intercept = false)
        β = path.betas[:]
        # update z - projection to constraint set
        v = β .+ u
        copy!(zold, z)
        z = proj(v)
        # update scaled dual variables u
        dualresnorm = norm((z - zold) / admmscale)
        primalres = β .- z
        primalresnorm = norm(primalres)
        u += primalres
        # check convergence criterion
        if (primalresnorm <= √p * admmabstol
                + admmreltol * max(vecnorm(β), vecnorm(z))) &&
                (dualresnorm <= √n * admmabstol
                + admmreltol * vecnorm(u / admmscale))
            break
        end
        # update ADMM scale parameter if requested
        if admmvaryscale
            if primalresnorm / dualresnorm > 10
                admmscale /= 2
                u /= 2
            elseif primalresnorm / dualresnorm < 0.1
                admmscale *= 2
                u *= 2
            end
        end
    end

    return β

end # end of function
