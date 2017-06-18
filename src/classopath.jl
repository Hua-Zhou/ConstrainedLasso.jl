# functions for constrained lasso path algorithm

export lsq_classopath

function lsq_classopath(
    X::AbstractMatrix,
    y::AbstractVector;
    Aeq::AbstractMatrix   = zeros(eltype(X), 0, size(X, 2)),
    beq::AbstractVector   = zeros(eltype(X), size(A, 1)),
    Aineq::AbstractMatrix = zeros(eltype(X), 0, size(X, 2)),
    bineq::AbstractVector = zeros(eltype(X), size(A, 1)),
    lb                    = typemin(eltype(X)),
    ub                    = typemax(eltype(X)),
    ρridge::Number        = zero(eltype(X))
    )

    T = promote_type(eltype(X), eltype(y))
    n, p = size(X)
    H = At_mul_B(X, X) + λridge * I

    # process lower bound and upper bound information
    if length(lb) == 1; lb = repmat([lb[1]], p); end
    if length(ub) == 1; lb = repmat([ub[1]], p); end
    if any(lb .> ub)
        error("conflict between lower bounds `lb` and upper bounds`ub`")
    end
    βposidx, βnegidx = Int(), Int(p)
    for i in 1:p
        if lb[i] > ub[i]
            error("conflict between lower bounds lb[$i] and upper bounds u[$i]")
        elseif lb[i] < ub[i]
            if lb[i] == 0
                push!(βposidx, i)
            elseif ub[i] == 0
                push!(βnegidx, 1)
            end
            if lb[i] > typemin(T) && lb[i] ≠ 0
                row = zeros(T, 1, p)
                row[i] = -1
                Aineq = [Aineq; row]
                bineq = [bineq; -lb[i]]
            elseif ub[i] < typmemax(T) && ub[i] ≠ 0
                row = zeros(T, 1, p)
                row[i] = 1
                Aineq = [Aineq; row]
                bineq = [bineq; ub[i]]
            end
        elseif lb[i] == ub[i]
            if lb[i] > typemin(T) && ub[i] < typemax(T)
                row = zeros(T, 1, p)
                row[i] = 1
                Aeq = [Aeq; row]
                beq = [beq; lb[i]]
            end
        end
    end

    # allocate variables along path
    neq, nineq = size(Aeq, 1), size(Aineq, 1)
    maxiters = 5(p + neq) # max number of path segments to consider
    βpath = zeros(p, maxiters)
    λpatheq = zeros(neq, maxiters) # dual variables for equality
    μpathineq = zeros(nineq, maxiters) # dual variables for inequality
    ρpath = zeros(maxiters) # tuning parameter
    dfpath = zeros(Int, maxiters) # degree of freedom
    objvalpath = zeros(maxiters) # objective value

    # initialization
    # solve LP to find solution at ρ = ∞
    β = Variable(p)
    

end
