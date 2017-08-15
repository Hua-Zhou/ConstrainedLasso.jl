module ConstrainedLasso

using Convex, SCS, GLMNet

# package code goes here
include("constrsparsereg.jl")
include("classopath.jl")
include("constrsparsereg_admm.jl")

export lsq_classopath, lsq_constrsparsereg, lsq_constrsparsereg_admm

end # module
