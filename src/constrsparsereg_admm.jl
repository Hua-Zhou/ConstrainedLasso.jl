"""
```
  lsq_constrsparsereg_admm(
    X             :: AbstractMatrix{T},
    y             :: AbstractVector{T},
    ρ             :: Number = zero(T);
    proj          :: Function = x -> x,
    obswt         :: Vector{T} = ones(T, length(y)),
    penwt         :: Vector{T} = ones(T, size(X, 2)),
    β0            :: Vector{T} = zeros(T, size(X, 2)),
    admmmaxite    :: Int = 10000,
    admmabstol    :: Float64 = 1e-4,
    admmreltol    :: Float64 = 1e-4,
    admmscale     :: Float64 = 1 / length(y),
    admmvaryscale :: Bool = false
    )
```
Fit constrained lasso at a fixed tuning parameter value by applying the
alternating direction method of multipliers (ADMM) algorithm.

### Arguments
- `X`       : predictor matrix.
- `y`       : response vector.
- `ρ`       : tuning parameter. Default 0.

### Optional arguments
- `proj`         : projection onto the constraint set. Default is identity (no constraint).
- `obswt`        : observation weights. Default is `[1 1 1 ... 1]`.
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
    y::AbstractVector{T},
    ρ::Number        = zero(T);
    proj::Function   = x -> x,
    obswt::Vector{T} = ones(T, length(y)),
    penwt::Vector{T} = ones(T, size(X, 2)),
    β0::Vector{T}    = zeros(T, size(X, 2)),
    admmmaxite::Int       = 10000,
    admmabstol::Float64   = 1e-4,
    admmreltol::Float64   = 1e-4,
    admmscale::Float64    = 1 / length(y),
    admmvaryscale::Bool   = false
    ) where T <: Union{Float64}

    n, p = size(X)
    β = copy(β0)
    z = proj(β)
    u = β - z
    v = similar(β)
    zold = similar(z)
    primalres = similar(z)

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
        copy!(β, path.betas)
        # update z - projection to constraint set
        v .= β .+ u
        copy!(zold, z)
        z = proj(v) ## could not do in-place; z = proj!(v) not working
        # update scaled dual variables u
        dualresnorm = norm((z - zold) / admmscale)
        # dualresnorm = 0.0
        # for j in 1:p
        #   dualresnorm += ((z[j] - zold[j]) / admmscale)^2
        # end
        # dualresnorm ^= 1/2
        primalres .= β .- z
        primalresnorm = norm(primalres)
        u .+= primalres
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


"""
```
  lsq_constrsparsereg_admm(
    X             :: AbstractMatrix{T},
    y             :: AbstractVector{T},
    ρlist         :: Vector;
    proj          :: Function = x -> x,
    obswt         :: Vector{T} = ones(T, length(y)),
    penwt         :: Vector{T} = ones(T, size(X, 2)),
    admmmaxite    :: Int = 10000,
    admmabstol    :: Float64 = 1e-4,
    admmreltol    :: Float64 = 1e-4,
    admmscale     :: Float64 = 1 / length(y),
    admmvaryscale :: Bool = false
    )
```
Fit constrained lasso at fixed tuning parameter values by applying the
alternating direction method of multipliers (ADMM) algorithm.

### Arguments
- `X`       : predictor matrix.
- `y`       : response vector.
- `ρlist`   : a vector of tuning parameter values.

### Optional arguments
- `proj`         : projection onto the constraint set. Default is identity (no constraint).
- `obswt`        : observation weights. Default is `[1 1 1 ... 1]`.
- `penwt`        : predictor penalty weights. Default is `[1 1 1 ... 1]`.
- `admmmaxite`   : maximum number of iterations for ADMM. Default is `10000`.
- `admmabstol`   : absolute tolerance for ADMM.
- `admmreltol`   : relative tolerance for ADMM.
- `admmscale`    : ADMM scale parameter. Default is `1/n`.
- `admmvaryscale`: dynamically chance the ADMM scale parameter. Default is false.

### Returns
- `βpath`       : estimated coefficents along the grid of `ρ` values.
"""

function lsq_constrsparsereg_admm(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    ρlist;
    proj::Function   = x -> x,
    obswt::Vector = ones(T, length(y)),
    penwt::Vector = ones(T, size(X, 2)),
    admmmaxite::Int       = 10000,
    admmabstol::Float64   = 1e-4,
    admmreltol::Float64   = 1e-4,
    admmscale::Float64    = 1 / length(y),
    admmvaryscale::Bool   = false
    ) where T

    βpath = zeros(T, size(X, 2), length(ρlist))
    for i in eachindex(ρlist)
      βpath[:, i] = lsq_constrsparsereg_admm(X, y, ρlist[i]; proj = proj,
            obswt = obswt, penwt = penwt, admmmaxite = admmmaxite,
            admmabstol = admmabstol, admmreltol = admmreltol,
            admmscale = admmscale, admmvaryscale = admmvaryscale)
    end
    return βpath
end
