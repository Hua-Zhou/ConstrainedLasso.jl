using SparseRegression, Convex, Mosek

srand(123)

n = 120
p = 20
X = randn(n, p)
β = ones(p)
y = X * β + randn(n)
obswt = ones(n)
penwt = ones(p)
λ = 1.0

obs = Obs(X, y, obswt)
s = SModel(obs, L1Penalty(), LinearRegression(), (λ / n) .* penwt)
o = learn!(s, ProxGrad(obs), MaxIter(500), Converged(coef; tol = 1e-8))

β̂ = Convex.Variable(p)
problem = Convex.minimize(0.5sumsquares(y - X * β̂) + λ * sumabs(β̂))
solve!(problem, MosekSolver(LOG=1))
@show [β̂.value o.β]
