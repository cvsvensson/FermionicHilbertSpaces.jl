# FermionicHilbertSpaces.jl

Symbolic fermions, bosons and spins, Hilbert spaces with constraints such as conserved particle number or parity, and the tensor products, partial traces and matrix representations that connect them. The package builds matrices; eigensolvers and time evolution come from LinearAlgebra or other packages.

The core workflow is `@fermions f` → `hilbert_space(f, labels, constraint)` → `representation(op, H, :dense)` → LinearAlgebra.

For task-by-task snippets (sector ground states, expectation values, moving vectors between a sector and the full space, operators between sectors, reduced density matrices, negativity, time evolution), read the cheat sheet: [docs/src/cheatsheet.md](docs/src/cheatsheet.md), also at https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/cheatsheet/.

Pitfalls:
- Use `representation(op, H, :dense)` for a dense matrix, not `Matrix(representation(op, H))`.
- Use `embed(ψ, Hsec => H)` and `indices(Hsec, H)` rather than padding vectors or filtering indices by hand.
- An operator that maps states out of the space errors unless `projection=true`, which drops those terms.
- Parity `1` is even and `-1` is odd; `ParityConservation()` lists the odd sector first.
- `exp` needs a matrix, not a symbolic operator.
