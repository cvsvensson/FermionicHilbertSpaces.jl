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
    "examples/floquet_tutorial.jl",
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
    "Home" => "index.md",
    "Cheat sheet" => "cheatsheet.md",
    "Tutorials" => [
        "Interacting Kitaev chain" => "literate_output/kitaev_chain.md",
        "Bosons" => "literate_output/bosons.md",
        "Spins" => "literate_output/spin_chain.md",
        "Mixed fermion-spin-boson system" => "mixed_fermion_spin_boson.md",
        "Open systems" => "literate_output/open_system_lindblad.md",
        "Entanglement and negativity" => "literate_output/partial_transpose.md",
        "SYK model" => "literate_output/syk_model.md",
        "Braiding Majorana zero modes" => "literate_output/braiding.md",
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
        "Internal functions" => "internals.md",
        "Internal types" => "internal_types.md",
    ],
    "API reference" => [
        "Public API" => "docstrings.md",
        "Free fermions API" => "free_fermions_api.md",
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
optional = ["Extending", "API reference"]
sections = [
    "Docs" => filter(p -> p.second isa String, pages),
    [p for p in pages if p.second isa Vector && p.first ∉ optional]...,
    "Optional" => [q for p in pages if p.first ∈ optional for q in p.second],
]
link((title, file)) = "- [$title]($(file == "index.md" ? "./" : replace(file, r"\.md$" => "/")))"
section((name, ps)) = "## $name\n\n" * join(link.(ps), "\n")
write(
    joinpath(@__DIR__, "build", "llms.txt"), """
    # FermionicHilbertSpaces.jl

    > Julia package for fermionic, bosonic and spin Hilbert spaces with constraints and symmetry sectors, and their sparse, dense or lazy operator matrices.

    Typical workflow: define symbols with `@fermions`/`@bosons`/`@spins`, build a space with `hilbert_space` (optionally with a constraint such as `NumberConservation`), get matrices with `representation(op, H)`, and reduce with `partial_trace`. Pass `:dense` to `representation` only when a dense matrix is needed.

    $(join(section.(sections), "\n\n"))
    """
)

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
