module ConstrainedLasso

using Convex, GLMNet, ECOS

# package code goes here
include("constrsparsereg.jl")
include("classopath.jl")
include("constrsparsereg_admm.jl")
#include("genlasso.jl")

export lsq_classopath, lsq_constrsparsereg, lsq_constrsparsereg_admm #, genlasso

end # module
