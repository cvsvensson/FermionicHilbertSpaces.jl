# Cheat sheet

Common tasks and the functions that do them. Snippets build on the variables of earlier ones (`H`, `Hsec`, `ψ`, `Ψ`, `HA`, `ham`); run them in order.

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
Conserve each species separately with a product of constraints; sectors are labelled by vectors such as `[1, 1]`.
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
expectation(f[1]'f[1], ψ, Hsec)
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
Rows are on `Hout` and columns on `Hin`. Here `f[1]'` from two to three particles. `Axes` and `dissipator` are public, not exported, so import them explicitly. An operator that maps states out of the space errors unless `projection=true`, which drops those terms.
```@example cheat
import FermionicHilbertSpaces: Axes
representation(f[1]', Axes(sector(3, H), Hsec))
```

## Reduced density matrix and entropy
```@example cheat
HA = hilbert_space(f, 1:2)
ρA = partial_trace(ψ, Hsec => subregion(HA, Hsec))
-sum(λ * log(λ) for λ in eigvals(Hermitian(ρA)) if λ > 0)
```

## Partial transpose and negativity
`logarithmic_negativity` is not exported, so import it.
```@example cheat
import FermionicHilbertSpaces: logarithmic_negativity
logarithmic_negativity(Ψ, H, HA)
```
```@example cheat
ρ = Ψ * Ψ'
partial_transpose(ρ, H, HA)
```

## Basis states and product states
`basisstates` lists the basis in the order of the representations; `Kets` parses a label into a state.
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
Evolve with the eigendecomposition of a sector, or pass the sparse `representation` to an ODE solver.
```@example cheat
ψ0 = representation(Kets(Hsec)("1100"), Hsec)
ψt(t) = vecs * (cis.(-vals * t) .* (vecs' * ψ0))
ψt(1.0) ≈ exp(-im * representation(ham, Hsec, :dense)) * ψ0
```

## Open system
`open_system` doubles the space for vectorized density matrices. See [Open systems](literate_output/open_system_lindblad.md).
```@example cheat
import FermionicHilbertSpaces: dissipator
Hlr, Hl, Hr, left, right = open_system(f, 1:2)
L = -im * (left(f[1]'f[2] + hc) - right(f[1]'f[2] + hc)) + dissipator(0.1 * f[1])
representation(L, Hlr)
```

## Large spaces
`:lazy` gives a matrix-free operator for iterative eigensolvers such as Arpack or KrylovKit, which must be installed separately.
```@example cheat
using Arpack
Hbig = hilbert_space(f, 1:12, NumberConservation(6))
hbig = representation(sum(f[n+1]'f[n] + hc for n in 1:11) + 0.5 * sum(f[n]'f[n] * f[n+1]'f[n+1] for n in 1:11), Hbig, :lazy)
real(eigs(hbig; nev=1, which=:SR)[1][1])
```
