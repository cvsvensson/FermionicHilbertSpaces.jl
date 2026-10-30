using FermionicHilbertSpaces

flatten_pages(pages) = reduce(vcat, (p.second isa AbstractString ? [p] : flatten_pages(p.second) for p in pages))

function write_llms(
        pages; builddir = joinpath(@__DIR__, "build"), srcdir = joinpath(@__DIR__, "src"),
        base = "https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/"
    )
    flat = flatten_pages(pages)
    url(file) = base * (file == "index.md" ? "" : replace(file, r"\.md$" => "/"))
    summary = "Julia package for fermionic, bosonic and spin Hilbert spaces with constraints and symmetry sectors, and their sparse, dense or lazy operator matrices."
    links = join(("- [$title]($(url(file)))" for (title, file) in flat), "\n")
    write(joinpath(builddir, "llms.txt"), "# FermionicHilbertSpaces.jl\n\n> $summary\n\n## Docs\n\n$links\n")
    pagetext = join(("# $title\n\n" * read(joinpath(srcdir, file), String) for (title, file) in flat), "\n\n")
    docs = join(
        (
            "## `$name`\n\n" * string(Base.Docs.doc(Base.Docs.Binding(m, name)))
                for m in (FermionicHilbertSpaces, FermionicHilbertSpaces.FreeFermions) for name in sort!(names(m)) if name != nameof(m)
        ), "\n\n"
    )
    return write(joinpath(builddir, "llms-full.txt"), "# FermionicHilbertSpaces.jl\n\n> $summary\n\n$pagetext\n\n# Docstrings\n\n$docs\n")
end
