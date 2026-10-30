# To build docs with LiveServer and avoid infinite loop with literate, run `servedocs(skip_dir = "docs/src/literate_output")`
using FermionicHilbertSpaces
using Documenter
import DocumenterCodeBlocks
import Literate

DocMeta.setdocmeta!(FermionicHilbertSpaces, :DocTestSetup, :(using FermionicHilbertSpaces); recursive=true)

literate_files = ["examples/kitaev_chain.jl", "examples/free_fermions.jl", "examples/floquet_tutorial.jl", "examples/open_system_lindblad.jl", "examples/spin_chain.jl", "examples/bosons.jl", "examples/clock_operators.jl", "examples/translation_symmetry.jl"]
output_directory = "docs/src/literate_output"
rm(output_directory; recursive = true, force = true)
mkpath(output_directory)
for file in literate_files
    Literate.markdown(file, output_directory; documenter=true, execute=false)
end

makedocs(;
    modules = [FermionicHilbertSpaces, FermionicHilbertSpaces.FreeFermions],
    authors="Viktor Svensson, William Samuelson",
    sitename="FermionicHilbertSpaces.jl",
    format=Documenter.HTML(;
        canonical="https://cvsvensson.github.io/FermionicHilbertSpaces.jl",
        edit_link="main",
        assets=String[],
        size_threshold = 400 * 2^10,
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Interacting Kitaev chain" => "literate_output/kitaev_chain.md",
            "Bosons" => "literate_output/bosons.md",
            "Spins" => "literate_output/spin_chain.md",
            "Mixed fermion-spin-boson system" => "mixed_fermion_spin_boson.md",
            "Open systems" => "literate_output/open_system_lindblad.md",
            "Spatial symmetries" => "literate_output/translation_symmetry.md",
        ],
        "Guides" => [
            "Constrained spaces and quantum numbers" => "conservation.md",
            "Hilbert space operations" => "hilbert_space_operations.md",
            "Symbolic states" => "symbolic_states.md",
            "Fermionic tensor products and partial traces" => "fermions.md",
            "Non-interacting systems" => "non_interacting.md",
            "Example: Free fermions" => "literate_output/free_fermions.md",
        ],
        "Performance" => [
            "Parallelization" => "parallelization.md",
        ],
        "Extending" => [
            "Extending and performance hooks" => "extending.md",
            "Custom operators: Clock" => "literate_output/clock_operators.md",
            "Custom operators: Floquet" => "literate_output/floquet_tutorial.md",
        ],
        "API reference" => "docstrings.md",
    ],
    plugins=[DocumenterCodeBlocks.CodeBlocks()],
)

deploydocs(;
    repo="github.com/cvsvensson/FermionicHilbertSpaces.jl",
    devbranch="main",
    push_preview=true
)
