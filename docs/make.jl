using FermionicHilbertSpaces
using Documenter
using Literate

DocMeta.setdocmeta!(FermionicHilbertSpaces, :DocTestSetup, :(using FermionicHilbertSpaces); recursive=true)

literate_files = ["examples/kitaev_chain.jl"]
output_directory = "docs/src/literate_output"
for file in literate_files
    Literate.markdown(file, output_directory; documenter=true, execute=false)
end

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
        "Tutorials" => ["Interacting Kitaev chain" => "literate_output/kitaev_chain.md",]
    ],
)

deploydocs(;
    repo="github.com/cvsvensson/FermionicHilbertSpaces.jl",
    devbranch="main",
    push_preview=true
)
