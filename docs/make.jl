# To build docs with LiveServer and avoid infinite loop with literate, run `servedocs(skip_dir = "docs/src/literate_output")`
using FermionicHilbertSpaces
using Documenter
import DocumenterCodeBlocks
import Literate

DocMeta.setdocmeta!(FermionicHilbertSpaces, :DocTestSetup, :(using FermionicHilbertSpaces); recursive=true)

literate_files = ["examples/kitaev_chain.jl", "examples/free_fermions.jl", "examples/floquet_tutorial.jl", "examples/open_system_lindblad.jl", "examples/spin_chain.jl", "examples/bosons.jl", "examples/clock_operators.jl"]
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
        "Fermionic tensor products and partial traces" => "fermions.md",
        "Constrained Hilbert spaces and conserved quantum numbers" => "conservation.md",
        "Hilbert space and array operations" => "hilbert_space_operations.md",
        "Tutorials and examples" => [
            "Interacting Kitaev chain" => "literate_output/kitaev_chain.md",
            "Bosons" => "literate_output/bosons.md",
            "Spins" => "literate_output/spin_chain.md",
            "Mixed fermion-spin-boson system" => "mixed_fermion_spin_boson.md",
            "Open systems" => "literate_output/open_system_lindblad.md",
            "Symbolic states" => "symbolic_states.md",
            "Custom operators: Clock" => "literate_output/clock_operators.md",
            "Custom operators: Floquet" => "literate_output/floquet_tutorial.md",
        ],
        "Non-interacting systems" => [
            "non_interacting.md" => "Non-interacting hilbert spaces",
            "literate_output/free_fermions.md" => "Example: Free fermions"],
        "Misc" => "misc.md",
        "Functions" => "docstrings.md",
    ],
    plugins=[DocumenterCodeBlocks.CodeBlocks()],
)

deploydocs(;
    repo="github.com/cvsvensson/FermionicHilbertSpaces.jl",
    devbranch="main",
    push_preview=true
)
