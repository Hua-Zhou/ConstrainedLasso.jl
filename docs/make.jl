push!(LOAD_PATH, "../src/")

using Documenter, ConstrainedLasso

makedocs(
      format = :html,
      sitename = "ConstrainedLasso",
      pages = Any[
          "Home" => "index.md",
          "Examples" => Any[
#              "demo/data1/.md",
#              "demo/data2/.md",
              "demo/data3/microbiome.md",
              "demo/lasso/demo_lasso.md",
          ]
      ]

)

deploydocs(
      repo    = "github.com/Hua-Zhou/ConstrainedLasso.jl.git",
      target  = "build",
      deps    = nothing,
      make    = nothing
)
