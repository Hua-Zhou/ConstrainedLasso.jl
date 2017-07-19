module ConstrainedLasso

using Convex, Mosek, SCS

# package code goes here
include("constrsparsereg.jl")
include("classopath.jl")

export lsq_classopath, lsq_constrsparsereg

end # module
