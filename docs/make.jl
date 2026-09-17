# To build docs with LiveServer and avoid infinite loop with literate, run `servedocs(skip_dir = "docs/src/literate_output")`
using FermionicHilbertSpaces
using Documenter
using Literate

DocMeta.setdocmeta!(FermionicHilbertSpaces, :DocTestSetup, :(using FermionicHilbertSpaces); recursive=true)

literate_files = ["examples/kitaev_chain.jl", "examples/free_fermions.jl", "examples/floquet_tutorial.jl", "examples/open_system_lindblad.jl", "examples/spin_chain.jl", "examples/bose_hubbard.jl"]
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
        "Fermions" => "fermions.md",
        "Conserved quantities" => "conservation.md",
        "Tutorials" => [
            "Bosons" => "literate_output/bose_hubbard.md",
            "Spins" => "literate_output/spin_chain.md",
            "Custom algebra: Floquet" => "literate_output/floquet_tutorial.md",
            "Open systems" => "literate_output/open_system_lindblad.md",
            "Symbolic states" => "symbolic_states.md",
            "Non interacting systems" => "non_interacting.md",
        ],
        "Examples" => [
            "Interacting Kitaev chain" => "literate_output/kitaev_chain.md",
            "Free fermions" => "literate_output/free_fermions.md",
            "Mixed fermion-spin-boson system" => "mixed_fermion_spin_boson.md"
        ],
        "Misc" => "misc.md",
        "Functions" => "docstrings.md",
    ],
)

deploydocs(;
    repo="github.com/cvsvensson/FermionicHilbertSpaces.jl",
    devbranch="main",
    push_preview=true
)
