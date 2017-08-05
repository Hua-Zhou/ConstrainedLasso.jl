using Documenter, ConstrainedLasso

ENV["DOCUMENTER_DEBUG"] = "true"

makedocs()

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "mkdocs-material", "python-markdown-math"),
    repo   = "github.com:Hua-Zhou/ConstrainedLasso.jl.git",
    julia  = "0.6",
    osname = "osx"
)
