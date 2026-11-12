# To build docs with LiveServer and avoid infinite loop with literate, run `servedocs(skip_dir = "docs/src/literate_output")`
using FermionicHilbertSpaces
using Documenter
import DocumenterCodeBlocks
import Literate

DocMeta.setdocmeta!(FermionicHilbertSpaces, :DocTestSetup, :(using FermionicHilbertSpaces); recursive=true)

# One entry per line, sorted, so PRs adding examples rarely conflict.
literate_files = [
    "examples/bosons.jl",
    "examples/braiding.jl",
    "examples/clock_operators.jl",
    "examples/fermions_tutorial.jl",
    "examples/floquet_tutorial.jl",
    "examples/fragmentation.jl",
    "examples/free_fermions.jl",
    "examples/kitaev_chain.jl",
    "examples/open_system_lindblad.jl",
    "examples/partial_transpose.jl",
    "examples/spin_chain.jl",
    "examples/syk_model.jl",
    "examples/translation_symmetry.jl",
]
output_directory = "docs/src/literate_output"
rm(output_directory; recursive = true, force = true)
mkpath(output_directory)
for file in literate_files
    Literate.markdown(file, output_directory; documenter=true, execute=false)
end

pages = [
    "Welcome page" => "index.md",
    "Cheat sheet" => "cheatsheet.md",
    "Tutorials" => [
        "Fermions" => "literate_output/fermions_tutorial.md",
        "Bosons" => "literate_output/bosons.md",
        "Spins" => "literate_output/spin_chain.md",
        "Mixed fermion-spin-boson system" => "mixed_fermion_spin_boson.md",
    ],
    "How-to guides" => [
        "Constrained spaces and quantum numbers" => "conservation.md",
        "Spatial symmetries" => "literate_output/translation_symmetry.md",
        "Hilbert-space fragmentation and quantum many-body scars" => "literate_output/fragmentation.md",
        "Hilbert space and array operations" => "hilbert_space_operations.md",
        "Entanglement and negativity" => "literate_output/partial_transpose.md",
        "Symbolic bras and kets" => "symbolic_states.md",
        "Open systems" => "literate_output/open_system_lindblad.md",
        "Non-interacting systems" => "non_interacting.md",
    ],
    "Examples" => [
        "Interacting Kitaev chain" => "literate_output/kitaev_chain.md",
        "Braiding Majorana zero modes" => "literate_output/braiding.md",
        "SYK model" => "literate_output/syk_model.md",
        "Free fermions in 2D" => "literate_output/free_fermions.md",
    ],
    "Explanation" => [
        "How the package works" => "how_it_works.md",
        "Fermionic tensor products, partial traces, and partial transpose" => "fermions.md",
    ],
    "Performance" => [
        "Performance tips" => "performance.md",
        "Parallelization" => "parallelization.md",
        "Benchmarking" => "benchmarking.md",
    ],
    "Extending" => [
        "Extending the package" => "extending.md",
        "Custom operators: Clock" => "literate_output/clock_operators.md",
        "Custom operators: Floquet" => "literate_output/floquet_tutorial.md",
    ],
    "Reference" => [
        "Public API" => "docstrings.md",
        "Free fermions API" => "free_fermions_api.md",
        "Internal functions" => "internals.md",
        "Internal types" => "internal_types.md",
    ],
]

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
    pages,
    plugins=[DocumenterCodeBlocks.CodeBlocks()],
)

# llms.txt (https://llmstxt.org/) is served next to the pages, so relative links resolve per docs version.
optional = ["Extending", "Reference"]
sections = [
    "Docs" => filter(p -> p.second isa String, pages),
    [p for p in pages if p.second isa Vector && p.first ∉ optional]...,
    "Optional" => [q for p in pages if p.first ∈ optional for q in p.second],
]
link((title, file)) = "- [$title]($(file == "index.md" ? "./" : replace(file, r"\.md$" => "/")))"
section((name, ps)) = "## $name\n\n" * join(link.(ps), "\n")
#! format: off
write(
    joinpath(@__DIR__, "build", "llms.txt"), """
    # FermionicHilbertSpaces.jl

    > Julia package for fermionic, bosonic and spin Hilbert spaces with constraints and symmetry sectors, and their sparse, dense or lazy operator matrices.

    Typical workflow: define symbols with `@fermions`/`@bosons`/`@spins`, build a space with `hilbert_space` (optionally with a constraint such as `NumberConservation`), get matrices with `representation(op, H)`, and reduce with `partial_trace`. Use `representation(op, H, Matrix)` only when a dense matrix is needed.

    $(join(section.(sections), "\n\n"))
    """
)
#! format: on

# Concurrent PR previews race to push gh-pages; each retry re-fetches it.
for attempt in 1:3
    try
        deploydocs(;
            repo = "github.com/cvsvensson/FermionicHilbertSpaces.jl",
            devbranch = "main",
            push_preview = true
        )
        break
    catch
        attempt == 3 && rethrow()
        sleep(15 * attempt)
    end
end
