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

## Spinful fermions
Labels can be any values, such as tuples. Conserve each spin species separately with a product of constraints; sectors are then labelled by vectors such as `[1, 1]`, not tuples.
```@example cheat
@fermions c
Hup = hilbert_space(c, [(i, :↑) for i in 1:2])
Hdn = hilbert_space(c, [(i, :↓) for i in 1:2])
Hσ = tensor_product(Hup, Hdn; constraint = NumberConservation(Hup) * NumberConservation(Hdn))
sector([1, 1], Hσ)
```

## Bosons, spins and mixed spaces
The second argument of `hilbert_space` is the number of levels for a boson and the spin for a spin. `tensor_product` combines any spaces.
```@example cheat
@boson b
@spins S 1//2
Hb = hilbert_space(b, 3)
HS = hilbert_space(S, 1:2)
Hm = tensor_product(Hup, Hb, HS)
representation(b'b * S[1][:z] + c[1, :↑]' * b + hc, Hm)
```

## Ground state of a sector
```@example cheat
Hsec = sector(2, H)
vals, vecs = eigen(Hermitian(representation(ham, Hsec, :dense)))
ψ = vecs[:, 1]
```

## Loop over sectors
```@example cheat
[(q, dim(Hq)) for (q, Hq) in zip(quantumnumbers(H), sectors(H))]
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

## Basis states and product states
`basisstates` lists the basis in the order of the vector and matrix representations. `Kets` parses a label into a state; on a product space, pass one label per factor.
```@example cheat
basisstates(Hsec)
```
```@example cheat
state_index("1100", Hsec)
```
```@example cheat
representation(Kets(Hsec)("1100"), Hsec)
```

## Time evolution
The package does not evolve states, but a few lines of LinearAlgebra do. `exp` needs a matrix, not a symbolic operator.
```@example cheat
ψ0 = representation(Kets(Hsec)("1100"), Hsec)
ψt(t) = vecs * (cis.(-vals * t) .* (vecs' * ψ0))
ψt(1.0) ≈ exp(-im * representation(ham, Hsec, :dense)) * ψ0
```
For Lindblad evolution, see [Open systems](literate_output/open_system_lindblad.md).

## Large spaces
`:lazy` gives a matrix-free operator for iterative eigensolvers such as Arpack or KrylovKit.
```@example cheat
using Arpack
Hbig = hilbert_space(f, 1:12, NumberConservation(6))
hbig = representation(sum(f[n+1]'f[n] + hc for n in 1:11) + 0.5 * sum(f[n]'f[n] * f[n+1]'f[n+1] for n in 1:11), Hbig, :lazy)
eigs(hbig; nev=1, which=:SR)[1]
```

## Pitfalls
- Use `representation(op, H, :dense)` for a dense matrix, not `Matrix(representation(op, H))`.
- Use `embed` and `indices` rather than padding vectors or filtering indices by hand.
- An operator that maps states out of the space errors unless `projection=true`, which drops those terms.
- `partial_trace(x, H => Hsub)` maps to a smaller space and `embed(x, Hsub => H)` back to the larger one.
