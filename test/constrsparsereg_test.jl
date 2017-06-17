info("Test lsq_constrsparsereg")

srand(123)
n, p = 100, 20
# truth with sum constraint sum(β) = 0
β = zeros(p)
β[1:round(Int, p / 4)] = 0
β[(round(Int, p / 4) + 1):round(Int, p / 2)] = 1
β[(round(Int, p / 2) + 1):round(Int, 3p / 4)] = 0
β[(round(Int, 3p / 4) + 1):p] = -1
# generate data
X = randn(n, p)
y = X * β + randn(n)
# penalty parameter
λ = 10.0
A = ones(1, p)
sense = '='
b = 0.0
penwt  = ones(p)
solver = SCSSolver(verbose=1)
#solver = ECOSSolver()
#solver = MosekSolver(LOG=1)
#solver = GurobiSolver(OutputFlag=1)
# fit using quadprog
β̂, objval, = lsq_constrsparsereg(X, y, λ; A = A, sense = sense, b = b,
    penwt = penwt, solver = solver)
@test_approx_eq_eps sum(β̂) 0.0 1e-6
