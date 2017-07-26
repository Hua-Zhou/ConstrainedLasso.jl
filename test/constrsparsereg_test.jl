module constrsparsereg_test

using Base.Test, ConstrainedLasso, SCS

info("Test lsq_constrsparsereg: sum-to-zero constraint")

# set up
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
solver = SCSSolver(verbose=0)
#solver = ECOSSolver()
# using Mosek; solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8);
#solver = GurobiSolver(OutputFlag=1)

info("Optimize at a single tuning parameter values")
ρ = 10.0
β̂, = lsq_constrsparsereg(X, y, ρ; Aeq = Aeq, beq = beq,
    penwt = penwt, solver = solver)
@test sum(β̂)≈0.0 atol=1e-5


info("Optimize at multiple tuning parameter values")
ρlist = 1.0:10.0
β̂, = lsq_constrsparsereg(X, y, ρlist; Aeq = Aeq, beq = beq,
    penwt = penwt, solver = solver)
@testset "zero-sum for multiple param values" begin
for si in sum(β̂, 1)
  @test si≈0.0 atol=1.0e-5
end
end


end # constrsparsereg_test module
