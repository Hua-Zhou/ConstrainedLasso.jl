"""
```
  lsq_classopath(
    X      :: AbstractMatrix{T},
    y      :: AbstractVector{T};
    Aeq    :: AbstractMatrix = zeros(T, 0, size(X, 2)),
    beq    :: Union{AbstractVector, Number} = zeros(T, size(Aeq, 1)),
    Aineq  :: AbstractMatrix = zeros(T, 0, size(X, 2)),
    bineq  :: Union{AbstractVector, Number} = zeros(T, size(Aineq, 1)),
    ρridge :: Number = zero(T),
    penidx :: Array{Bool} = fill(true, size(X, 2)),
    solver = ECOSSolver(maxit=10e8, verbose=0)
    )
```
Calculate the solution path of the constrained lasso problem that minimizes
    `0.5sumabs2(√obswt .* (y - X * β)) + ρ * sumabs(penwt .* β)`
subject to linear constraints.

### Arguments
- `X`       : predictor matrix.
- `y`       : response vector.

### Optional arguments
- `Aeq`     : equality constraint matrix.
- `beq`     : equality constraint vector.
- `Aineq`   : inequality constraint matrix.
- `bineq`   : inequality constraint vector.
- `ρridge`  : tuning parameter for ridge penalty. Default is 0.
- `penidx`  : a logical vector indicating penalized coefficients.
- `solver`  : a solver Convex.jl supports. Default is ECOS.
              Note that Mosek and Gurobi are more robust than ECOS. Unlike ECOS or
              SCS, both Mosek and Gurobi require a license (free for academic
              use). For details, see <http://convexjl.readthedocs.io/en/latest/solvers.html>.
### Examples
See tutorial examples at <https://hua-zhou.github.io/ConstrainedLasso.jl/latest/demo/path/>.
"""

function lsq_classopath(
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    Aeq::AbstractMatrix   = zeros(T, 0, size(X, 2)),
    beq::Union{AbstractVector, Number} = zeros(T, size(Aeq, 1)),
    Aineq::AbstractMatrix = zeros(T, 0, size(X, 2)),
    bineq::Union{AbstractVector, Number} = zeros(T, size(Aineq, 1)),
    ρridge::Number        = zero(T),
    penidx::Array{Bool}   = fill(true, size(X, 2)),
    solver = ECOSSolver(maxit=10e8, verbose=0)
    ) where T

    # T = promote_type(eltype(X), eltype(y))
    n, p = size(X)

    if n < p
        warn("Adding a small ridge penalty (default is 1e-4) since n < p")
        if ρridge <= 0
            warn("ρridge must be positive, switching to default value (1e-4)")
            ρridge = 1e-4
        end
        # create augmented data
        y = [y; zeros(p)]
        X = [X; √ρridge * eye(p)]
        # record original number of observations
        # n_orig = n
    else
        # make sure X is full column rank
        R = qrfact(X)[:R]
        rankX = sum(abs.(diag(R)) .> abs(R[1,1]) * max(n,p) * eps(eps(eltype(R))))

        if rankX != p
            warn("Adding a small ridge penalty (default is 1e-4) since X is rank deficient")
            if ρridge <= 0
                warn("ρridge must be positive, switching to default value (1e-4)")
                ρridge = 1e-4
            end
            # create augmented data
            y = [y; zeros(p)]
            X = [X; √ρridge * eye(p)]
        end
    end


    # allocate variables along path
    neq, nineq = size(Aeq, 1), size(Aineq, 1)
    maxiters = 5(p + nineq)               # max number of path segments to consider
    βpath = zeros(p, maxiters)
    λpatheq = zeros(neq, maxiters)      # dual variables for equality
    μpathineq = zeros(nineq, maxiters)  # dual variables for inequality
    ρpath = zeros(maxiters)             # tuning parameter
    dfpath = fill(Inf, maxiters)        # degree of freedom
    objvalpath = zeros(maxiters)        # objective value
    violationspath = fill(Inf, maxiters)

    ### initialization
    # use LP to find ρmax
    H = At_mul_B(X, X)
    #sense = [repmat(['='], neq); repmat(['<'], nineq)]
    #β, _, problem = lsq_constrsparsereg(X, y, Inf;      # why necessary?
    #          [Aeq; Aineq], sense, [beq; bineq], lb, ub)#

    # find the maximum ρ (starting value)
    ρpath[1], idx, = find_ρmax(X, y; Aeq = Aeq, beq = beq, Aineq = Aineq,
              bineq = bineq, solver = solver)

    # calculate at ρmax
    βpath[:, 1], objvalpath[1], problem = lsq_constrsparsereg(X, y, ρpath[1];
              Aeq = Aeq, beq = beq, Aineq = Aineq, bineq = bineq,
              penwt = penidx, solver = solver);

    for i in 1:min(2, length(problem.constraints))
      if eval((problem.constraints[i]).head) == ==
        λpatheq[:, 1] = problem.constraints[i].dual
      elseif eval((problem.constraints[i]).head) == <=
        μpathineq[:, 1] = problem.constraints[i].dual
      end
    end

    μpathineq[μpathineq .< 0] = 0
    setActive = (abs.(βpath[:, 1]) .> 1e-4) .| (.~penidx)
    βpath[find(.!setActive), 1] = 0

    residIneq = Aineq * βpath[:, 1] - bineq
    setIneqBorder = residIneq .== 0
    nIneqBorder = countnz(setIneqBorder)

    # initialize subgradient vector
    resid = y - X * βpath[:, 1]
    subgrad = X' * resid - Aeq' * λpatheq[:, 1] - Aineq' * μpathineq[:, 1]

    subgrad[setActive] = sign.(βpath[find(setActive), 1])
    subgrad[.!setActive] = subgrad[.!setActive] / ρpath[1]
    setActive[idx] = true
    nActive = countnz(setActive)

    # calculate degrees of freedom
    rankAeq = rank(Aeq) ### Brian's comment: need to make it more efficient
    dfpath[1] = nActive - rankAeq - nIneqBorder

    # set initial violations counter to 0
    violationspath[1] = 0

    # sign for path direction (originally went both ways, but increasing was retired)
    dirsgn = -1

    ####################################
    ### main loop for path following ###
    ####################################

    k = 0

    for k in 2:maxiters

      # threshold near-zero rhos to zero and stop algorithm
      if ρpath[k-1] <= 1e-4
        ρpath[k-1] = 0
        break
      end

      # calculate derivative for coefficients and multipliers
      # construct matrix

      activeCoeffs = find(setActive)
      inactiveCoeffs = find(.!setActive)
      idxIneqBorder = find(setIneqBorder)

      M = hcat(H[activeCoeffs, activeCoeffs], Aeq[:, activeCoeffs]',
        Aineq[find(setIneqBorder), activeCoeffs]')
      M = vcat(M, zeros(neq + nIneqBorder, size(M, 2)))
      M[(end - neq - nIneqBorder + 1):end, 1:nActive] = [Aeq[:, activeCoeffs];
                Aineq[idxIneqBorder, activeCoeffs]]

      # calculate derivative
      dir = try
        dirsgn * (M \ [subgrad[setActive]; zeros(neq + nIneqBorder)])
      catch
        dir = -(pinv(M) * [subgrad[setActive]; zeros(neq + nIneqBorder)])
      end

      if any(isnan.(dir))
        dir = -(pinv(M) * [subgrad[setActive]; zeros(neq + nIneqBorder)])
      end

      # calculate the derivative for rho * subgradient
      dirSubgrad = - hcat(H[inactiveCoeffs, activeCoeffs], Aeq[:, inactiveCoeffs]',
                Aineq[idxIneqBorder, inactiveCoeffs]') * dir

      ### check additional events related to potential subgraient violations

      ## inactive coefficients moving too slowly
      # negative subgradient
      inactSlowNegIdx = find(((1*dirsgn - 1e-8) .<= subgrad[.~setActive]) .&
        (subgrad[.~setActive] .<= (1*dirsgn + 1e-8)) .&
        (1*dirsgn .< dirSubgrad))

      # positive subgradient
      inactSlowPosIdx = find(((-1*dirsgn - 1e-8) .<= subgrad[.~setActive]) .&
          (subgrad[.~setActive] .<= (-1*dirsgn + 1e-8)) .&
          (dirSubgrad .< -1*dirsgn))

      ## "active" coeficients estimated as 0 with potential sign mismatch #%
      # positive subgrad but negative derivative
      signMismatchPosIdx = find(((0 - 1e-8) .<= subgrad[setActive]) .&
          (subgrad[setActive] .<= (1 + 1e-8)) .&
          (dirsgn * dir[1:nActive] .<= (0 - 1e-8)) .&
          (βpath[activeCoeffs, k-1] .== 0))

      # Negative subgradient but positive derivative
      signMismatchNegIdx = find(((-1 - 1e-8) .<= subgrad[setActive]) .&
          (subgrad[setActive] .<= (0 + 1e-8)) .&
          ((0 + 1e-8) .<= dirsgn * dir[1:nActive]) .&
          (βpath[activeCoeffs, k-1] .== 0))

      # reset violation counter (to avoid infinite loops)
      violateCounter = 0

      # outer while loop for checking all conditions together
      while ( ~isempty(inactSlowNegIdx) || ~isempty(inactSlowPosIdx) ||
            ~isempty(signMismatchPosIdx) || ~isempty(signMismatchNegIdx) )

          # monitor and fix condition 1 violations
          while !isempty(inactSlowNegIdx)
            ## Identify and move problem coefficient
            # indices corresponding to inactive coefficients
            inactiveCoeffs = find(.!setActive)
            # identify problem coefficient
            viol_coeff = inactiveCoeffs[inactSlowNegIdx]
            setActive[viol_coeff] = true

            # determine new number of active coefficients
            nActive = countnz(setActive)
            # determine number of active/binding inequality constraints
            nIneqBorder = countnz(setIneqBorder)

            activeCoeffs = find(setActive)
            inactiveCoeffs = find(.!setActive)
            idxIneqBorder = find(setIneqBorder)
            M = hcat(H[activeCoeffs, activeCoeffs], Aeq[:, activeCoeffs]',
              Aineq[find(setIneqBorder), activeCoeffs]')
            M = vcat(M, zeros(neq + nIneqBorder, size(M, 2)))
            M[(end - neq - nIneqBorder + 1):end, 1:nActive] = [Aeq[:, activeCoeffs];
                      Aineq[idxIneqBorder, activeCoeffs]]

            # calculate derivative
            dir = try
              dirsgn * (M \ [subgrad[setActive]; zeros(neq + nIneqBorder)])
            catch
              dir = -(pinv(M) * [subgrad[setActive]; zeros(neq + nIneqBorder)])
            end

            ## calculate derivative for rho*subgradient #%
            dirSubgrad = - hcat(H[inactiveCoeffs, activeCoeffs], Aeq[:, inactiveCoeffs]',
                                    Aineq[idxIneqBorder, inactiveCoeffs]') * dir

            ## Misc. housekeeping #%
            # check for violations again

            ## inactive coefficients moving too slowly
            # negative subgradient
            inactSlowNegIdx = find(((1*dirsgn - 1e-8) .<= subgrad[.~setActive]) .&
              (subgrad[.~setActive] .<= (1*dirsgn + 1e-8)) .&
              (1*dirsgn .< dirSubgrad))

            # positive subgradient
            inactSlowPosIdx = find(((-1*dirsgn - 1e-8) .<= subgrad[.~setActive]) .&
                (subgrad[.~setActive] .<= (-1*dirsgn + 1e-8)) .&
                (dirSubgrad .< -1*dirsgn))

            ## "active" coeficients estimated as 0 with potential sign mismatch #%
            # positive subgrad but negative derivative
            signMismatchPosIdx = find(((0 - 1e-8) .<= subgrad[setActive]) .&
                (subgrad[setActive] .<= (1 + 1e-8)) .&
                (dirsgn * dir[1:nActive] .<= (0 - 1e-8)) .&
                (βpath[activeCoeffs, k-1] .== 0))

            # Negative subgradient but positive derivative
            signMismatchNegIdx = find(((-1 - 1e-8) .<= subgrad[setActive]) .&
                (subgrad[setActive] .<= (0 + 1e-8)) .&
                ((0 + 1e-8) .<= dirsgn * dir[1:nActive]) .&
                (βpath[activeCoeffs, k-1] .== 0))


            # update violation counter
            violateCounter = violateCounter + 1;
            # break loop if needed
            if violateCounter >= maxiters
                break
            end

          end

          # Monitor & fix subgradient condition 2 violations
          while ~isempty(inactSlowPosIdx)
              # Identify & move problem coefficient #%
              # indices corresponding to inactive coefficients
              inactiveCoeffs = find(.!setActive)
              # identify problem coefficient
              viol_coeff = inactiveCoeffs[inactSlowPosIdx]
              # put problem coefficient back into active set;
              setActive[viol_coeff] = true
              # determine new number of active coefficients
              nActive = countnz(setActive)
              # determine number of active/binding inequality constraints
              nIneqBorder = countnz(setIneqBorder)

              # Recalculate derivative for coefficients & multiplier #%
              # construct matrix

              activeCoeffs = find(setActive)
              inactiveCoeffs = find(.!setActive)
              idxIneqBorder = find(setIneqBorder)
              M = hcat(H[activeCoeffs, activeCoeffs], Aeq[:, activeCoeffs]',
                Aineq[find(setIneqBorder), activeCoeffs]')
              M = vcat(M, zeros(neq + nIneqBorder, size(M, 2)))
              M[(end - neq - nIneqBorder + 1):end, 1:nActive] = [Aeq[:, activeCoeffs];
                        Aineq[idxIneqBorder, activeCoeffs]]



              # calculate derivative
              dir = try
                dirsgn * (M \ [subgrad[setActive]; zeros(neq + nIneqBorder)])
              catch
                dir = -(pinv(M) * [subgrad[setActive]; zeros(neq + nIneqBorder)])
              end

              # calculate derivative for rho*subgradient #
              dirSubgrad = - hcat(H[inactiveCoeffs, activeCoeffs], Aeq[:, inactiveCoeffs]',
                                        Aineq[idxIneqBorder, inactiveCoeffs]') * dir

              # Misc. housekeeping #%
              # check for violations again
              inactSlowPosIdx = find(((-1*dirsgn - 1e-8) .<= subgrad[.!setActive]) .&
                  (subgrad[.!setActive] .<= (-1*dirsgn + 1e-8)) .&
                  (dirSubgrad .< -1*dirsgn))

              # "Active" coeficients est'd as 0 with potential sign mismatch #%
              # Positive subgrad but negative derivative
              signMismatchPosIdx = find(((0 - 1e-8) .<= subgrad[setActive]) .&
                  (subgrad[setActive] .<= (1 + 1e-8)) .&
                  (dirsgn * dir[1:nActive] .<= (0 - 1e-8)) .&
                  (βpath[activeCoeffs, k-1] .== 0))

              # Negative subgradient but positive derivative
              signMismatchNegIdx = find(((-1 - 1e-8) .<= subgrad[setActive]) .&
                  (subgrad[setActive] .<= (0 + 1e-8)) .&
                  ((0 + 1e-8) .<= dirsgn * dir[1:nActive]) .&
                  (βpath[activeCoeffs, k-1] .== 0))

              # update violation counter
              violateCounter = violateCounter + 1;
              # break loop if needed
              if violateCounter >= maxiters
                  break
              end
          end

          # Monitor & fix condition 3 violations
          while ~isempty(signMismatchPosIdx)
              ## Identify & move problem coefficient #%
              # indices corresponding to active coefficients
              activeCoeffs = find(setActive)
              # identify prblem coefficient
              viol_coeff = activeCoeffs[signMismatchPosIdx]
              # put problem coefficient back into inactive set;
              setActive[viol_coeff] = false
              # determine new number of active coefficients
              nActive = countnz(setActive)
              # determine number of active/binding inequality constraints
              nIneqBorder = countnz(setIneqBorder)

              ## Recalculate derivative for coefficients & multipliers #%
              # construct matrix
              activeCoeffs = find(setActive)
              inactiveCoeffs = find(.!setActive)
              idxIneqBorder = find(setIneqBorder)
              M = hcat(H[activeCoeffs, activeCoeffs], Aeq[:, activeCoeffs]',
                Aineq[idxIneqBorder, activeCoeffs]')
              M = vcat(M, zeros(neq + nIneqBorder, size(M, 2)))
              M[(end - neq - nIneqBorder + 1):end, 1:nActive] = [Aeq[:, activeCoeffs];
                        Aineq[idxIneqBorder, activeCoeffs]]

              # calculate derivative
              dir = try
                dirsgn * (M \ [subgrad[setActive]; zeros(neq + nIneqBorder)])
              catch
                dir = -(pinv(M) * [subgrad[setActive]; zeros(neq + nIneqBorder)])
              end

              # calculate derivative for rho*subgradient (Eq. 10) #%
              dirSubgrad = - hcat(H[inactiveCoeffs, activeCoeffs], Aeq[:, inactiveCoeffs]',
                                        Aineq[idxIneqBorder, inactiveCoeffs]') * dir


              ## Misc. housekeeping #%
              # check for violations again
              signMismatchPosIdx = find(((0 - 1e-8) .<= subgrad[setActive]) .&
                      (subgrad[setActive] .<= (1 + 1e-8)) .&
                      (dirsgn * dir[1:nActive] .<= (0 - 1e-8)) .&
                      (βpath[activeCoeffs, k-1] .== 0))

              # Negative subgradient but positive derivative
              signMismatchNegIdx = find(((-1 - 1e-8) .<= subgrad[setActive]) .&
                      (subgrad[setActive] .<= (0 + 1e-8)) .&
                      ((0 + 1e-8) .<= dirsgn * dir[1:nActive]) .&
                      (βpath[activeCoeffs, k-1] .== 0))

              # update violation counter
              violateCounter = violateCounter + 1
              # break loop if needed
              if violateCounter >= maxiters
                  break
              end
          end

          # Monitor & fix condition 4 violations
          while ~isempty(signMismatchNegIdx)
              ## Identify & move problem coefficient #%
              # indices corresponding to active coefficients
              activeCoeffs = find(setActive)
              # identify prblem coefficient
              viol_coeff = activeCoeffs[signMismatchNegIdx]
              # put problem coefficient back into inactive set;
              setActive[viol_coeff] = false
              # determine new number of active coefficients
              nActive = countnz(setActive)
              # determine number of active/binding inequality constraints
              nIneqBorder = countnz(setIneqBorder)

              ## Recalculate derivative for coefficients & multipliers #%
              # construct matrix
              activeCoeffs = find(setActive)
              inactiveCoeffs = find(.!setActive)
              idxIneqBorder = find(setIneqBorder)
              M = hcat(H[activeCoeffs, activeCoeffs], Aeq[:, activeCoeffs]',
                Aineq[idxIneqBorder, activeCoeffs]')
              M = vcat(M, zeros(neq + nIneqBorder, size(M, 2)))
              M[(end - neq - nIneqBorder + 1):end, 1:nActive] = [Aeq[:, activeCoeffs];
                        Aineq[idxIneqBorder, activeCoeffs]]

              # calculate derivative
              dir = try
                dirsgn * (M \ [subgrad[setActive]; zeros(neq + nIneqBorder)])
              catch
                dir = -(pinv(M) * [subgrad[setActive]; zeros(neq + nIneqBorder)])
              end


              # calculate derivative for rho*subgradient #%
              dirSubgrad = - hcat(H[inactiveCoeffs, activeCoeffs], Aeq[:, inactiveCoeffs]',
                                        Aineq[idxIneqBorder, inactiveCoeffs]') * dir


              # Recheck for violations #%
              signMismatchNegIdx = find(((-1 - 1e-8) .<= subgrad[setActive]) .&
                        (subgrad[setActive] .<= (0 + 1e-8)) .&
                        ((0 + 1e-8) .<= dirsgn * dir[1:nActive]) .&
                        (βpath[activeCoeffs, k-1] .== 0))
              # update violation counter
              violateCounter = violateCounter + 1
              # break loop if needed
              if violateCounter >= maxiters
                  break
              end

          end

          ## update violation trackers to see if any issues persist ##%

          activeCoeffs = find(setActive)
          inactiveCoeffs = find(.!setActive)
          idxIneqBorder = find(setIneqBorder)

          # Inactive coefficients moving too slowly #%
          # Negative subgradient
          inactSlowNegIdx =
              find(((1 * dirsgn - 1e-8) .<= subgrad[.!setActive]) .&
              (subgrad[.~setActive] .<= (1 * dirsgn + 1e-8)) .&
              (1 * dirsgn .< dirSubgrad))
%         # Positive subgradient
          inactSlowPosIdx = find(((-1*dirsgn - 1e-8) .<= subgrad[.!setActive]) .&
              (subgrad[.!setActive] .<= (-1 * dirsgn + 1e-8)) .&
              (dirSubgrad .< -1 * dirsgn))

          # "Active" coefficients estimated as 0 with potential sign mismatch #%
          # Positive subgrad but negative derivative
          signMismatchPosIdx = find(((0 - 1e-8) .<= subgrad[setActive]) .&
              (subgrad[setActive] .<= (1 + 1e-8)) .&
              (dirsgn * dir[1:nActive] .<= (0 - 1e-8)) .&
              (βpath[activeCoeffs, k-1] .== 0))

          # Negative subgradient but positive derivative
          signMismatchNegIdx = find(((-1 - 1e-8) .<= subgrad[setActive]) .&
              (subgrad[setActive] .<= (0 + 1e-8)) .&
              ((0 + 1e-8) .<= dirsgn * dir[1:nActive]) .&
              (βpath[activeCoeffs, k-1] .== 0))

          # break loop if needed
          if violateCounter >= maxiters
              break
          end
      end # end of outer while loop


      # store number of violations
      violationspath[k] = violateCounter;

      # calculate derivative for residual inequality
      activeCoeffs = find(setActive)
      inactiveCoeffs = find(.!setActive)
      idxIneqBorder = find(setIneqBorder)

      dirResidIneq = Aineq[find(.!setIneqBorder), activeCoeffs] * dir[1:nActive]

      ### Determine rho for next event (via delta rho) ###%
      ## Events based on coefficients changing activation status ##%

      # clear previous values for delta rho
      nextρβ = fill(Inf, p, 1)

      # Active coefficient going inactive #%
      nextρβ[setActive] = -dirsgn * βpath[activeCoeffs, k-1] ./ dir[1:nActive]

      # Inactive coefficient becoming positive #%
      t1 = dirsgn * ρpath[k-1] * (1 - subgrad[inactiveCoeffs]) ./ (dirSubgrad - 1)
      # threshold values hitting ceiling
      t1[t1 .<= 1e-8] = Inf;

      # Inactive coefficient becoming negative #%
      t2 = -dirsgn * ρpath[k-1] * (1 + subgrad[.!setActive]) ./ (dirSubgrad + 1)
      # threshold values hitting ceiling
      t2[t2 .<= 1e-8] = Inf

      # choose smaller delta rho out of t1 and t2
      nextρβ[.!setActive] = min.(t1, t2)
      # ignore delta rhos numerically equal to zero
      nextρβ[(nextρβ .<= 1e-8) .| (.!penidx)] = Inf


      ## Events based inequality constraints ##%
	    # clear previous values
      nextρIneq = fill(Inf, nineq, 1)

      # Inactive inequality constraint becoming active #%
      nextρIneq[.!setIneqBorder] = reshape(-dirsgn * residIneq[.!setIneqBorder],
                countnz(.!setIneqBorder), 1) ./
                reshape(dirResidIneq, countnz(.~setIneqBorder), 1)

      # Active inequality constraint becoming deactive #%
      if !isempty(μpathineq)
      nextρIneq[setIneqBorder] =
      - dirsgn * μpathineq[idxIneqBorder, k-1] ./
                reshape(dir[nActive+neq+1:end], nIneqBorder, 1)
      end

      # ignore delta rhos equal to zero
      nextρIneq[nextρIneq .<= 1e-8] = Inf

      ## determine next rho ##
      # find smallest rho
      chgρ = findmin([nextρβ; nextρIneq])[1]
      # find all indices corresponding to this chgρ
      idx = find(([nextρβ; nextρIneq] - chgρ) .<= 1e-8)

      # terminate path following if no new event found
      if chgρ == Inf
         chgρ = ρpath[k-1]
      end


      ## Update values at new rho ##%
      # move to next rho #%
      # make sure next rho isn't negative
      if ρpath[k-1] + dirsgn * chgρ < 0
          chgρ = ρpath[k-1]
      end
      # calculate new value of rho
      ρpath[k] = ρpath[k-1] + dirsgn * chgρ

      ## Update parameter and subgradient values #%
      # new coefficient estimates
      activeCoeffs = find(setActive)

      βpath[activeCoeffs, k] = βpath[activeCoeffs, k-1] +
            dirsgn * chgρ * dir[1:nActive]
      # force near-zero coefficients to be zero (helps with numerical issues)
      βpath[abs.(βpath[:, k]) .< 1e-12, k] = 0

      # new subgradient estimates
      subgrad[.!setActive] = (ρpath[k-1] * subgrad[.!setActive] +
            dirsgn * chgρ * dirSubgrad) / ρpath[k]

      # Update dual variables #%
      # update lambda (lagrange multipliers for equality constraints)
      if !isempty(λpatheq)
        λpatheq[:, k] = λpatheq[:, k-1] +
            dirsgn * chgρ * reshape(dir[nActive + 1:nActive + neq], neq, 1)
      end
      # update mu (lagrange multipliers for inequality constraints)
      if !isempty(μpathineq)
        μpathineq[idxIneqBorder, k] = μpathineq[idxIneqBorder, k-1] +
            dirsgn * chgρ * reshape(dir[nActive + neq + 1:end], nIneqBorder, 1)
      end
      # update residual inequality
      residIneq = Aineq * βpath[:, k] - bineq


      ## update sets ##%
      for j = 1:length(idx)
          curidx = idx[j]
          if curidx <= p && setActive[curidx]
              # an active coefficient hits 0, or
              setActive[curidx] = false
          elseif curidx <= p && ~setActive[curidx]
              # a zero coefficient becomes nonzero
              setActive[curidx] = true
          elseif curidx > p
              # an ineq on boundary becomes strict, or
              # a strict ineq hits boundary
              setIneqBorder[curidx - p] = !setIneqBorder[curidx - p]
          end
      end

      # determine new number of active coefficients
      nActive = countnz(setActive)
      # determine number of active/binding inequality constraints
      nIneqBorder = countnz(setIneqBorder)

      ## Calcuate and store values of interest along the path #%
      # calculate value of objective function
      objvalpath[k] = norm(y - X * βpath[:, k])^2/2 +
            ρpath[k] * sum(abs.(βpath[:, k]))

      # calculate degrees of freedom
      dfpath[k] = nActive - rankAeq - nIneqBorder
      # break algorithm when df are exhausted
      if dfpath[k] >= n
          break
      end

    end # end of big for loop

       βpath = βpath[:, 1:k-1]
       deleteat!(ρpath, k:length(ρpath))
       deleteat!(objvalpath, k:length(objvalpath))
       deleteat!(dfpath, k:length(dfpath))
       dfpath[dfpath .< 0] = 0;


    return βpath, ρpath, objvalpath, λpatheq, μpathineq, dfpath, violationspath

end # end of the function

"""
```
  find_ρmax(
    X      :: AbstractMatrix,
    y      :: AbstractVector;
    Aeq    :: AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    beq    :: Union{AbstractVector, Number} = zeros(eltype(X), size(Aeq, 1)),
    Aineq  :: AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    bineq  :: Union{AbstractVector, Number} = zeros(eltype(X), size(Aineq, 1)),
    penidx :: Array{Bool} = fill(true, size(X, 2)),
    solver = ECOSSolver(maxit=10e8, verbose=0)
    )
```
Find the maximum tuning parameter value `ρmax` to kick-start the solution path.
"""
function find_ρmax(
    X::AbstractMatrix,
    y::AbstractVector;
    Aeq::AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    beq::Union{AbstractVector, Number} = zeros(eltype(X), size(Aeq, 1)),
    Aineq::AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    bineq::Union{AbstractVector, Number} = zeros(eltype(X), size(Aineq, 1)),
    penidx::Array{Bool} = fill(true, size(X, 2)),
    solver = ECOSSolver(maxit=10e8, verbose=0)
    )

    p = size(X, 2)

    x = Variable(p)
    problem = minimize(dot(ones(p), abs(x)), Aeq * x == beq,
              Aineq * x <= bineq)

    problem = minimize(dot(ones(eltype(X), size(X, 2)), abs(x)))
    if !isempty(Aeq)
      problem.constraints += Aeq * x == beq
    end
    if !isempty(Aineq)
      problem.constraints += Aineq * x <= bineq
    end

    # TT = STDOUT # save original STDOUT stream
    # redirect_stdout()
    solve!(problem, solver)
    β = x.value
    # redirect_stdout(TT) # restore STDOUT

    λeq = zeros(eltype(X), 0, 1)
    μineq = zeros(eltype(X), 0, 1)

    for i in 1:min(2, length(problem.constraints))
      if eval((problem.constraints[i]).head) == ==
        λeq = problem.constraints[i].dual
      elseif eval((problem.constraints[i]).head) == <=
        μineq = problem.constraints[i].dual
      end
    end

    if !isempty(μineq)
      μineq[μineq .< 0] = 0
    end
    setActive = (abs.(β) .> 1e-4) .| (.~penidx)
    β[.!setActive] = 0

    resid = y - X * β
    subgrad = X' * resid - Aeq' * λeq - Aineq' * μineq
    ρmax, indρmax = findmax(abs.(subgrad))

    return ρmax, indρmax, problem.optval, λeq, μineq
end
