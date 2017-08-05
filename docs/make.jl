using Documenter, ConstrainedLasso

ENV["DOCUMENTER_DEBUG"] = "true"

makedocs(
    doctest   = false,
    format    = :html,
    clean     = true,
    sitename  = "ConstrainedLasso.jl",
    modules   = [ConstrainedLasso],
    pages     = [
        "Home"     => "index.md",
        "References" => "references.md"
    ]
)

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "mkdocs-material", "python-markdown-math"),
    repo    = "github.com/Hua-Zhou/ConstrainedLasso.jl.git",
    julia   = "0.6",
    osname  = "osx"
)
