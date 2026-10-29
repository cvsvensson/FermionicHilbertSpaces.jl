# Cheat sheet

Common tasks and the functions that do them. All snippets continue from the first one.

## Space and Hamiltonian
`@fermions f` defines symbolic fermions. `hilbert_space(f, labels, constraint)` gives a space, and `representation(op, H, :dense)` its matrix.
```@example cheat
using FermionicHilbertSpaces, LinearAlgebra
@fermions f
N = 4
ham = sum(f[n+1]'f[n] + hc for n in 1:N-1) + sum(f[n]'f[n] * f[n+1]'f[n+1] for n in 1:N-1)
H = hilbert_space(f, 1:N, NumberConservation())
```

## Ground state of a sector
```@example cheat
Hsec = sector(2, H)
vals, vecs = eigen(Hermitian(representation(ham, Hsec, :dense)))
ψ = vecs[:, 1]
```

## Expectation value
```@example cheat
dot(ψ, representation(f[1]'f[1], Hsec), ψ)
```

## Sector vector in the full space
`embed` places the vector at the sector's indices and pads with zeros.
```@example cheat
Ψ = embed(ψ, Hsec => H)
```

## Indices of a sector in the full space
Use `indices` instead of filtering on the diagonal of a number operator.
```@example cheat
indices(Hsec, H) == indices(2, H)
```
```@example cheat
Ψ[indices(2, H)] == ψ
```

## Parity
Parity `1` is even and `-1` is odd. `ParityConservation()` lists the odd sector first.
```@example cheat
P = parityoperator(H)
dot(Ψ, P, Ψ)
```
```@example cheat
quantumnumbers(hilbert_space(f, 1:N, ParityConservation()))
```

## Operators between sectors
Build a space with both sectors, represent the operator with `projection=true` (dropping terms that leave the space), and slice with `indices`. Here `f[1]'` from two to three particles:
```@example cheat
Hmix = hilbert_space(f, 1:N, NumberConservation([2, 3]))
cdag = representation(f[1]', Hmix; projection=true)
cdag[indices(3, Hmix), indices(2, Hmix)]
```

## Reduced density matrix and entropy
`subregion(HA, Hsec)` keeps the states of `HA` that appear in `Hsec`. The state should have definite parity for the reduced density matrix to be physical.
```@example cheat
HA = hilbert_space(f, 1:2)
ρA = partial_trace(ψ, Hsec => subregion(HA, Hsec))
-sum(λ * log(λ) for λ in eigvals(Hermitian(ρA)) if λ > 0)
```

## Partial transpose
```@example cheat
ρ = Ψ * Ψ'
partial_transpose(ρ, H, HA)
```

## Time evolution
The package does not evolve states, but a few lines of LinearAlgebra do. `exp` needs a matrix, not a symbolic operator.
```@example cheat
ψ0 = zeros(ComplexF64, dim(Hsec))
ψ0[state_index("1100", Hsec)] = 1
ψt(t) = vecs * (cis.(-vals * t) .* (vecs' * ψ0))
ψt(1.0) ≈ exp(-im * representation(ham, Hsec, :dense)) * ψ0
```
For Lindblad evolution, see [Open systems](literate_output/open_system_lindblad.md).

## Pitfalls
- Use `representation(op, H, :dense)` for a dense matrix, not `Matrix(representation(op, H))`.
- Use `embed` and `indices` rather than padding vectors or filtering indices by hand.
- An operator that maps states out of the space errors unless `projection=true`, which drops those terms.
- `partial_trace(x, H => Hsub)` maps to a smaller space and `embed(x, Hsub => H)` back to the larger one.
