module admm_test

using Base.Test, ConstrainedLasso, GLMNet

info("Test lsq_constrsparsereg_admm: sum-to-zero constraint (single param value)")
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
# fit at a fixed parameter value
ρ = 2.0
β̂admm1 = lsq_constrsparsereg_admm(X, y, ρ; proj = x -> x - mean(x))
@test sum(β̂admm1)≈0.0 atol=1e-5

info("Test lsq_constrsparsereg_admm: sum-to-zero constraint (multiple param values)")
# fit at fixed parameter values
ρ = 1.0:2.0:20.0
β̂admm2 = lsq_constrsparsereg_admm(X, y, ρ; proj = x -> x - mean(x))
@testset "zero-sum for multiple param values" begin
for si in sum(β̂admm2, 1)
    @test si≈0.0 atol=1e-5
end
end

info("Test lsq_constrsparsereg_admm: non-negativity constraint")
# set up
n, p = 100, 20
β = zeros(p)
# truth
β[1:10] = 1:10
# generate data
srand(41)
X = randn(n, p)
y = X * β + randn(n)
# fit at the fixed parameter value; admmvaryscale=true
ρ = 3
β̂admm3 = lsq_constrsparsereg_admm(X, y, ρ; proj = x -> clamp.(x, 0, Inf),
        admmvaryscale = true)
@test all(β̂admm3 .>= 0)

end # end of module
