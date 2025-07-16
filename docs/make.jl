using FermionicHilbertSpaces
using Documenter
using Literate

DocMeta.setdocmeta!(FermionicHilbertSpaces, :DocTestSetup, :(using FermionicHilbertSpaces); recursive=true)

input_file = "examples/kitaev_chain.jl"
output_directory = "docs/src/examples"
Literate.markdown(input_file, output_directory; documenter=true, execute=false)
display(pwd())

makedocs(;
    modules=[FermionicHilbertSpaces],
    authors="Viktor Svensson, William Samuelson",
    sitename="FermionicHilbertSpaces.jl",
    format=Documenter.HTML(;
        canonical="https://cvsvensson.github.io/FermionicHilbertSpaces.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Misc" => "misc.md",
        "Functions" => "docstrings.md",
        "Tutorials" => ["Interacting Kitaev chain" => "kitaev_chain.md",]
    ],
)

deploydocs(;
    repo="github.com/cvsvensson/FermionicHilbertSpaces.jl",
    devbranch="main",
    push_preview=true
)
