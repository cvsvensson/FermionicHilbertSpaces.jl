using FermionicHilbertSpaces

flatten_pages(pages) = reduce(vcat, (p.second isa AbstractString ? [p] : flatten_pages(p.second) for p in pages))

function write_llms(pages; builddir=joinpath(@__DIR__, "build"), srcdir=joinpath(@__DIR__, "src"),
    base="https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/")
    flat = flatten_pages(pages)
    url(file) = base * (file == "index.md" ? "" : replace(file, r"\.md$" => "/"))
    summary = "Julia package for fermionic, bosonic and spin Hilbert spaces with constraints and symmetry sectors, and their sparse, dense or lazy operator matrices."
    links = join(("- [$title]($(url(file)))" for (title, file) in flat), "\n")
    write(joinpath(builddir, "llms.txt"), "# FermionicHilbertSpaces.jl\n\n> $summary\n\n## Docs\n\n$links\n")
    clean(txt) = replace(txt, r"^(`{3,})@(meta|setup|index|autodocs|raw)[^\n]*\n.*?^\1\r?\n"ms => "", r"^(`{3,})@(example|repl)[^\n]*"m => s"\1julia", r"\]\(@ref[^)]*\)" => "]")
    pagetext = join(("# $title\n\n" * clean(read(joinpath(srcdir, file), String)) for (title, file) in flat if file != "docstrings.md"), "\n\n")
    docstring(m, name) = try string(Base.Docs.doc(Base.Docs.Binding(m, name))) catch e; @warn "No docstring for $name" e; nothing end
    entries = ((name, docstring(m, name)) for m in (FermionicHilbertSpaces, FermionicHilbertSpaces.FreeFermions) for name in sort!(names(m)) if name != nameof(m))
    docs = join(("## `$name`\n\n$(clean(doc))" for (name, doc) in entries if !isnothing(doc)), "\n\n")
    write(joinpath(builddir, "llms-full.txt"), "# FermionicHilbertSpaces.jl\n\n> $summary\n\n$pagetext\n\n# Docstrings\n\n$docs\n")
end
