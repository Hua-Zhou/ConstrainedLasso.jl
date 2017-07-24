info("Test lsq_constrsparsereg")

### set up ###
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
Aeq = ones(1, p)
beq = [0.0]
penwt  = ones(p)
#solver = SCSSolver(verbose=0)
#solver = ECOSSolver()
using Mosek; solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8);
#solver = GurobiSolver(OutputFlag=1)

info("Optimize at a single tuning parameter values")
ρ = 10.0
β̂, = lsq_constrsparsereg(X, y, ρ; Aeq = Aeq, beq = beq,
    penwt = penwt, solver = solver)
#@show β̂
@test sum(β̂)≈0.0 atol=1e-6

info("Optimize at multiple tuning parameter values")

ρlist = 1.0:10.0
β̂, = lsq_constrsparsereg(X, y, ρlist; Aeq = Aeq, beq = beq,
    penwt = penwt, solver = solver)
#@show sum(β̂, 1)

@testset "zero-sum for multiple param values" begin for i in sum(β̂, 1)
  @test i≈0.0 atol=1.0e-6
end
end

println("passed")



info("Test lsq_classopath 1")
logging(DevNull, ConstrainedLasso, :lsq_classopath, kind=:warn) # hide
β̂path1, ρpath1, objpath, = lsq_classopath(X, y; Aeq = Aeq, beq = beq, solver = solver)
@test all(abs.(sum(β̂path1, 1)) .< 1e-6)


info("Test lsq_classopath 2")

### set up ###
n, p = 50, 100
# define true parameter vector: sparsity with a few non-zero coefficients
β = zeros(p)
β[1:10] = 1:10
# inequality constraints
Aineq = - eye(p)
bineq = zeros(p)
# generate data
srand(41)
X = randn(n, p)
y = X * β + randn(n)
using Mosek; solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8);

β̂path2, ρpath2, = lsq_classopath(X, y; Aineq = Aineq, bineq = bineq,
          solver = solver)
#@test all(β̂path2 .>= -0.5)


@testset "non-negativity" begin for i in reshape(β̂path2, p * length(ρpath2), 1)
  @test i >= 0
end
end

println("passed")
