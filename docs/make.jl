push!(LOAD_PATH, "../src/")

using Documenter, ConstrainedLasso

makedocs(
      doctest   = false,
      format    = :html,
      clean     = true,
      sitename  = "ConstrainedLasso.jl",
      modules   = [ConstrainedLasso],
      pages     = [
          "Home"     => "index.md",
          "Interface" => "interface.md",
          "Simulations" => "demo/sim.md",
          "Real Data Applications" => [
              "demo/prostate.md",
              "demo/warming.md",
              "demo/tumor.md",
              "demo/microbiome.md",
          ]
      ]

)

deploydocs(
      repo    = "github.com/Hua-Zhou/ConstrainedLasso.jl.git",
      target  = "build",
      julia   = "0.6",
      deps    = nothing,
      make    = nothing
)
