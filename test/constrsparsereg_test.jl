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
A = ones(1, p)
sense = '='
b = 0.0
penwt  = ones(p)
solver = SCSSolver(verbose=0)
#solver = ECOSSolver()
#using Mosek; solver = MosekSolver(LOG=1)
#solver = GurobiSolver(OutputFlag=1)

info("Optimize at a single tuning parameter values")
ρ = 10.0
β̂, = lsq_constrsparsereg(X, y, ρ; A = A, sense = sense, b = b,
    penwt = penwt, solver = solver)
@show β̂
@test_approx_eq_eps sum(β̂) 0.0 1e-6

info("Optimize at multiple tuning parameter values")

ρlist = 1.0:10.0
β̂ = lsq_constrsparsereg(X, y, ρlist; A = A, sense = sense, b = b,
    penwt = penwt, solver = solver)
println(β̂)
@show sum(β̂, 1)

#info("Test lsq_classopath")

#lsq_classopath(X, y; Aeq, beq, Aineq, Bineq, )
