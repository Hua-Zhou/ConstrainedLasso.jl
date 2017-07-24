module PkgTest

using Base.Test, ConstrainedLasso, SCS, ECOS

# write your own tests here
@testset "constrsparsereg" begin include("constrsparsereg_test.jl") end
# include("")
# include("")
#
# # List of tests
# tests = [
#
#        ]
#
# println("Running tests:")
#
# for t in tests
#   @testset "$(t)" begin
#     include("$(t).jl")
#   end
# end


end # end of module
