# Cheat sheet

Common tasks and the functions that do them. Snippets build on the variables of earlier ones; run them in order. Results are mostly suppressed with `;`, so run a line to see its output.

## Spaces
`@fermions f` defines symbolic fermions, `hilbert_space(f, labels, constraint)` a space, and `representation(op, H)` the sparse matrix of an operator (add `:dense` when you need a dense one, `:lazy` for a matrix-free one). Products of constraints conserve each species separately, and `tensor_product` combines any spaces, including bosons (levels given as the number) and spins (value set in the macro).
```@example cheat
using FermionicHilbertSpaces, LinearAlgebra
@fermions f
N = 4
ham = sum(f[n+1]'f[n] + hc for n in 1:N-1) + sum(f[n]'f[n] * f[n+1]'f[n+1] for n in 1:N-1)
H = hilbert_space(f, 1:N, NumberConservation())

@fermions c
@boson b
@spins S 1//2
Hup = hilbert_space(c, [(i, :↑) for i in 1:2])
Hdn = hilbert_space(c, [(i, :↓) for i in 1:2])
Hσ = tensor_product(Hup, Hdn; constraint = NumberConservation(Hup) * NumberConservation(Hdn))
Hσ1 = sector([1, 1], Hσ)
Hm = tensor_product(Hup, hilbert_space(b, 3), hilbert_space(S, 1:2))
representation(b'b * S[1][:z] + c[1, :↑]' * b + hc, Hm);
```
A subregion given as a space must list its modes in the order of the parent space: `subregion(hilbert_space(c, [(1, :↑), (1, :↓)]), Hσ)` is the space of site 1.

## Free fermions
A quadratic Hamiltonian on a `SingleParticle()` space is its single-particle matrix `h`, whose negative eigenvalues sum to the ground-state energy. `BdG()` includes pairing, where the energy is `tr(h)/2 - sum(filter(>(0), ε))` for the eigenvalues `ε` of the BdG matrix; see [Non-interacting hamiltonians](non_interacting.md).
```@example cheat
using FermionicHilbertSpaces.FreeFermions
h = Hermitian(rand(N, N))
Hs = hilbert_space(f, 1:N, SingleParticle())
ε = eigvals(Hermitian(representation(sum(f[i]' * h[i, j] * f[j] for i in 1:N, j in 1:N), Hs, :dense)))
sum(filter(<(0), ε))
```

## Sectors and states
`sector`, `sectors` and `quantumnumbers` select and loop over the blocks of a constrained space. `embed` places a sector vector in the full space, padding with zeros, and `indices` gives its positions there (use it instead of filtering on a number operator). Parity `1` is even and `-1` is odd, and `ParityConservation()` lists the odd sector first. `Axes(Hout, Hin)` gives the block between two sectors, with rows on `Hout`; an operator that leaves the space errors unless `projection=true`. `Axes` is public, not exported.
```@example cheat
import FermionicHilbertSpaces: Axes
Hsec = sector(2, H)
vals, vecs = eigen(Hermitian(representation(ham, Hsec, :dense)))
ψ = vecs[:, 1]
dims = [(q, dim(Hq)) for (q, Hq) in zip(quantumnumbers(H), sectors(H))];
expectation(f[1]'f[1], ψ, Hsec)
Ψ = embed(ψ, Hsec => H);
@assert Ψ[indices(2, H)] == ψ && indices(Hsec, H) == indices(2, H)
P = parityoperator(H)
dot(Ψ, P, Ψ)
quantumnumbers(hilbert_space(f, 1:N, ParityConservation()));
representation(f[1]', Axes(sector(3, H), Hsec));
```

## Subsystems
`partial_trace(ψ, Hsec => Hsub)` gives the reduced density matrix on a subregion, from which the entropy follows. `partial_transpose` and `logarithmic_negativity` (not exported, so import it) measure entanglement.
```@example cheat
import FermionicHilbertSpaces: logarithmic_negativity
HA = hilbert_space(f, 1:2)
ρA = partial_trace(ψ, Hsec => subregion(HA, Hsec))
-sum(λ * log(λ) for λ in eigvals(Hermitian(ρA)) if λ > 0)
logarithmic_negativity(Ψ, H, HA)
partial_transpose(Ψ * Ψ', H, HA);
```

## Basis states and dynamics
`basisstates` lists the basis in the order of the representations, and `Ket(label, H)` parses a label into a state. Evolve with the eigendecomposition of a sector, or pass the sparse `representation` to an ODE solver.
```@example cheat
basisstates(Hsec);
state_index("1100", Hsec);
ψ0 = representation(Ket("1100", Hsec), Hsec)
ψt(t) = vecs * (cis.(-vals * t) .* (vecs' * ψ0))
@assert ψt(1.0) ≈ exp(-im * representation(ham, Hsec, :dense)) * ψ0
```

## Open systems
`open_system` doubles the space for vectorized density matrices; `dissipator` is public, not exported. See [Open systems](literate_output/open_system_lindblad.md).
```@example cheat
import FermionicHilbertSpaces: dissipator
Hlr, Hl, Hr, left, right = open_system(f, 1:2)
L = -im * (left(f[1]'f[2] + hc) - right(f[1]'f[2] + hc)) + dissipator(0.1 * f[1])
representation(L, Hlr);
```

## Large spaces
`:lazy` gives a matrix-free operator for iterative eigensolvers such as Arpack or KrylovKit, which must be installed separately (the example needs `using Arpack`).
```@example cheat
using Arpack
Hbig = hilbert_space(f, 1:12, NumberConservation(6))
hbig = representation(sum(f[n+1]'f[n] + hc for n in 1:11) + 0.5 * sum(f[n]'f[n] * f[n+1]'f[n+1] for n in 1:11), Hbig, :lazy)
real(eigs(hbig; nev=1, which=:SR)[1][1])
```
