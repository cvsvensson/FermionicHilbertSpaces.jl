@testmodule PropertyGenerators begin
    using Supposition, FermionicHilbertSpaces, LinearAlgebra, Random
    export f, γ, b, S, polynomial, fermion_atoms, spin_atoms, seeds, draw, rand_density, ≊, CHECK

    @fermions f
    @majoranas γ
    @bosons b
    @spins S 1 // 2

    # A fresh seed each run explores new examples; failures print the shrunk counterexample.
    const CHECK = Supposition.CheckConfig(; rng = Xoshiro(rand(RandomDevice(), UInt)), max_examples = 100, db = false)

    coeffs = Data.SampledFrom([1, -1, 2, 1 // 2, 1im, 2 - 1im])
    monomial(atoms) = map((c, xs) -> c * prod(xs; init = 1), coeffs, Data.Vectors(Data.SampledFrom(atoms); max_size = 3))
    "Sums of up to three products of up to three of `atoms`, with small exact coefficients."
    polynomial(atoms) = map(sum, Data.Vectors(monomial(atoms); min_size = 1, max_size = 3))
    fermion_atoms(labels) = vcat([f[i] for i in labels], [f[i]' for i in labels])
    spin_atoms(s) = [s[o] for o in (:x, :y, :z)]
    seeds = Data.Integers{UInt32}()
    "Draw from `gen` inside a property; the value is listed under `name` in a counterexample."
    draw(name, gen) = (x = Data.produce!(gen); event!(name, x); x)

    ≊(a, b) = isapprox(a, b; atol = 1.0e-10)
    "A random density matrix on `H`, commuting with `P` if given."
    function rand_density(H, seed, P = I)
        X = randn(Xoshiro(seed), ComplexF64, dim(H), dim(H))
        ρ = X * X' + P * X * X' * P'
        ρ / tr(ρ)
    end
end

@testitem "Property: operator algebra" setup = [PropertyGenerators] begin
    using Supposition, Random
    p = polynomial(vcat(fermion_atoms(1:2), [γ[1], γ[2]], [b[1], b[1]'], spin_atoms(S[1])))
    @check config = CHECK function associative(x = p, y = p, z = p)
        (x * y) * z == x * (y * z)
    end
    @check config = CHECK function distributive(x = p, y = p, z = p)
        x * (y + z) == x * y + x * z && (x + y) * z == x * z + y * z
    end
    @check config = CHECK function adjoint_reverses(x = p, y = p)
        (x * y)' == y' * x' && iszero(x - x)
    end
    label = Data.Integers(1, 3)
    @check config = CHECK function car_and_clifford(i = label, j = label)
        f[i] * f[j]' + f[j]' * f[i] == (i == j) && iszero(f[i] * f[j] + f[j] * f[i]) &&
            γ[i] * γ[j] + γ[j] * γ[i] == 2 * (i == j)
    end

    # Reordering factors of different species, keeping the order within each species, gives an equal product
    @fermions c
    species = [fermion_atoms(1:2), [c[1], c[1]'], [γ[1], γ[2]], [b[1], b[1]'], [S[1][o] for o in (:z, :+, :-)]]
    atom = Data.SampledFrom([(k, x) for (k, xs) in enumerate(species) for x in xs])
    @check config = CHECK function commuting_species(xs = Data.Vectors(atom; max_size = 6), seed = seeds)
        queues = [last.(filter(x -> first(x) == k, xs)) for k in eachindex(species)]
        reordered = [popfirst!(queues[k]) for k in shuffle(Xoshiro(seed), first.(xs))]
        x, y = prod(last.(xs); init = 1), prod(reordered; init = 1)
        x == y && hash(x) == hash(y)
    end
end

@testitem "Property: commutation relations in representations" setup = [PropertyGenerators] begin
    using Supposition, LinearAlgebra
    using FermionicHilbertSpaces: SpinField
    rep(x, H) = representation(x, H, :dense)
    axes = Data.SampledFrom([(:x, :y, :z), (:y, :z, :x), (:z, :x, :y)])
    @check config = CHECK function spin_algebra(J = Data.SampledFrom([1 // 2, 1, 3 // 2]), a = axes)
        s = SpinField(:s, J)[1]
        x, y, z = map(o -> s[o], a)
        H = hilbert_space(s)
        x * y - y * x == 1im * z && rep(x * y - y * x, H) ≈ 1im * rep(z, H) &&
            rep(x * x + y * y + z * z, H) ≈ J * (J + 1) * I
    end
    @check config = CHECK function boson_ccr(n = Data.Integers(2, 4))
        M = rep(b[1] * b[1]' - b[1]' * b[1], hilbert_space(b[1], n))
        M[1:(n - 1), 1:(n - 1)] == I
    end
end

@testitem "Property: representation is a homomorphism" setup = [PropertyGenerators] begin
    using Supposition, LinearAlgebra
    rep(x, H) = representation(x, H, :dense)
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(γ, 1:2), hilbert_space(S[1]))
    p = polynomial(vcat(fermion_atoms(1:2), [γ[1], γ[2]], spin_atoms(S[1])))
    v = randn(ComplexF64, dim(H))
    @check config = CHECK function homomorphism(x = p, y = p)
        M = rep(x, H)
        rep(x * y, H) ≊ M * rep(y, H) && rep(x + y, H) ≊ M + rep(y, H) && rep(x', H) ≊ M' &&
            representation(x, H) ≊ M && representation(x, H, :lazy) * v ≊ M * v
    end

    # Only states far below the boson cutoff: the vacuum, for products of at most six factors
    Hb = tensor_product(hilbert_space(f, 1:2), hilbert_space(b[1], 7))
    pb = polynomial(vcat(fermion_atoms(1:2), [b[1], b[1]']))
    vac = rep(1, Hb)[:, 1]
    @check config = CHECK function boson_homomorphism(x = pb, y = pb)
        rep(x * y, Hb) * vac ≊ rep(x, Hb) * (rep(y, Hb) * vac)
    end

    hops = map((c, i, j) -> c * f[i]' * f[j], Data.SampledFrom([1, -1, 1im]), Data.Integers(1, 3), Data.Integers(1, 3))
    conserving = map(sum, Data.Vectors(map(prod, Data.Vectors(hops; min_size = 1, max_size = 2)); min_size = 1, max_size = 3))
    @check config = CHECK function number_sectors(x = conserving, y = conserving, k = Data.SampledFrom([0, 1, 2, 3, nothing]))
        Hn = hilbert_space(f, 1:3, isnothing(k) ? NumberConservation() : NumberConservation(k))
        rep(x * y, Hn) ≊ rep(x, Hn) * rep(y, Hn)
    end
end

@testitem "Property: partial trace, embedding and tensor product" setup = [PropertyGenerators] begin
    using Supposition, LinearAlgebra, Random
    rep(x, H; kw...) = representation(x, H, :dense; kw...)
    parity(n) = prod((1 - 2 * f[i]' * f[i] for i in 1:n); init = 1)
    subsystems = @composed function subsystem(N = Data.Integers(2, 4), seed = seeds)
        rng = Xoshiro(seed)
        (N, sort(randperm(rng, N)[1:rand(rng, 1:N)]))
    end
    constraints = Data.SampledFrom([NoSymmetry(), ParityConservation(), NumberConservation()])

    # Expectation values of operators on the subsystem agree before and after the partial trace
    @check config = CHECK function partial_trace_expectation(s = subsystems, qn = constraints, seed = seeds)
        N, sub = s
        x = draw("x", polynomial(fermion_atoms(sub)))
        H = hilbert_space(f, 1:N, qn)
        Hsub = subregion(hilbert_space(f, sub), H)
        ρ = rand_density(H, seed, rep(parity(N), H))
        ρsub = partial_trace(ρ, H => Hsub)
        tr(ρsub) ≈ 1 && ρsub ≊ ρsub' && eigmin(Hermitian(ρsub)) > -1.0e-10 &&
            tr(rep(x, Hsub; projection = true) * ρsub) ≊ tr(rep(x, H; projection = true) * ρ)
    end
    @check config = CHECK function embedding(s = subsystems)
        N, sub = s
        x = draw("x", polynomial(fermion_atoms(sub)))
        Hsub = hilbert_space(f, sub)
        embed(rep(x, Hsub), Hsub => hilbert_space(f, 1:N)) ≊ rep(x, hilbert_space(f, 1:N))
    end
    @check config = CHECK function mixed_partial_trace(bits = Data.Integers(1, 15), n = Data.Integers(2, 3), seed = seeds)
        mask = Bool.(digits(bits; base = 2, pad = 4))
        spaces = [hilbert_space(f, 1:1), hilbert_space(f, 2:2), hilbert_space(b[1], n), hilbert_space(S[1])]
        atoms = [fermion_atoms(1:1), fermion_atoms(2:2), [b[1], b[1]'], spin_atoms(S[1])]
        H = tensor_product(spaces)
        Hsub = count(mask) == 1 ? only(spaces[mask]) : tensor_product(spaces[mask])
        x = draw("x", polynomial(reduce(vcat, atoms[mask])))
        ρ = rand_density(H, seed, rep(parity(2), H))
        tr(rep(x, Hsub) * partial_trace(ρ, H => Hsub)) ≊ tr(rep(x, H) * ρ)
    end
    @check config = CHECK function majorana_partial_trace(s = subsystems, seed = seeds)
        N, pairs = s
        sub = reduce(vcat, [2k - 1, 2k] for k in pairs)
        x = draw("x", polynomial([γ[i] for i in sub]))
        H, Hsub = hilbert_space(γ, 1:2N), hilbert_space(γ, sub)
        ρ = rand_density(H, seed, parityoperator(H))
        tr(rep(x, Hsub) * partial_trace(ρ, H => Hsub)) ≊ tr(rep(x, H) * ρ)
    end

    @check config = CHECK function tensor_product_of_operators(n = Data.Integers(1, 2), m = Data.Integers(1, 2))
        HA, HB = hilbert_space(f, 1:n), hilbert_space(f, (n + 1):(n + m))
        H = tensor_product(HA, HB)
        x = draw("x", polynomial(fermion_atoms(1:n)))
        y = draw("y", polynomial(fermion_atoms((n + 1):(n + m))))
        tensor_product((rep(x, HA), rep(y, HB)), (HA, HB) => H) ≊ rep(x * y, H)
    end
    @check config = CHECK function reshape_roundtrip(n = Data.Integers(2, 3), qn = Data.SampledFrom([NoSymmetry(), ParityConservation()]), seed = seeds)
        H = hilbert_space(f, 1:n, qn)
        Hs = Tuple(hilbert_space(f, i:i) for i in 1:n)
        m = randn(Xoshiro(seed), ComplexF64, dim(H), dim(H))
        reshape(reshape(m, H => Hs), Hs => H) == m && reshape(reshape(m[:, 1], H => Hs), Hs => H) == m[:, 1]
    end
end

@testitem "Property: symbolic states and open systems" setup = [PropertyGenerators] begin
    using Supposition, LinearAlgebra, Random
    using FermionicHilbertSpaces: SpinField, open_system, dissipator
    rep(x, H) = representation(x, H, :dense)
    # (space, operator atoms, fermion parity) for fermions, a spin, or both
    spaces = @composed function space(n = Data.Integers(1, 2), J = Data.SampledFrom([1 // 2, 1, 3 // 2]), kind = Data.Integers(1, 3))
        s = SpinField(:s, J)[1]
        Hf, Hs = hilbert_space(f, 1:n), hilbert_space(s)
        P = prod((1 - 2 * f[i]' * f[i] for i in 1:n); init = 1)
        kind == 1 ? (Hf, fermion_atoms(1:n), P) :
            kind == 2 ? (Hs, spin_atoms(s), 1) :
            (tensor_product(Hf, Hs), vcat(fermion_atoms(1:n), spin_atoms(s)), P)
    end

    @check config = CHECK function operator_on_ket(s = spaces, seed = seeds)
        H, atoms, _ = s
        x = draw("x", polynomial(atoms))
        k = Kets(H)
        c = randn(Xoshiro(seed), ComplexF64, dim(H), 2)
        ψ, φ = (sum(c[i, j] * k[i] for i in 1:dim(H)) for j in 1:2)
        v, w, M = representation(ψ, H), representation(φ, H), rep(x, H)
        representation(x * ψ, H) ≊ M * v && dot(φ, x, ψ) ≊ dot(w, M, v)
    end

    # Superoperators act on even density matrices as the matrix formulas
    @check config = CHECK function lindblad_dissipator(s = spaces, seed = seeds)
        H, atoms, P = s
        L = draw("L", polynomial(atoms))
        Hlr, Hl, Hr = open_system(H)
        ρ = rand_density(H, seed, rep(P, H))
        Lm = rep(L, H)
        dρ = reshape(rep(dissipator(L), Hlr) * reshape(ρ, (Hl, Hr) => Hlr), Hlr => (Hl, Hr))
        dρ ≊ Lm * ρ * Lm' - (Lm' * Lm * ρ + ρ * Lm' * Lm) / 2 && abs(tr(dρ)) < 1.0e-10
    end
    @check config = CHECK function left_right(s = spaces, seed = seeds)
        H, atoms, P = s
        A = draw("A", polynomial(atoms))
        B = draw("B", polynomial(atoms))
        Hlr, Hl, Hr, left, right = open_system(H)
        ρ = rand_density(H, seed, rep(P, H))
        out = reshape(rep(left(A) * right(B), Hlr) * reshape(ρ, (Hl, Hr) => Hlr), Hlr => (Hl, Hr))
        out ≊ rep(A, H) * ρ * rep(B, H)
    end
end

@testitem "Property: free-fermion ground states" setup = [PropertyGenerators] begin
    using Supposition, LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions
    # Random quadratic hamiltonians with pairing, against exact diagonalization
    @check config = CHECK function bdg_ground_state(n = Data.Integers(1, 3))
        quadratic = map(*, Data.SampledFrom([1, -1, 1 // 2, 1im]), Data.SampledFrom(fermion_atoms(1:n)), Data.SampledFrom(fermion_atoms(1:n)))
        x = draw("x", map(sum, Data.Vectors(quadratic; min_size = 1, max_size = 4)))
        ham = x + x' + sum(draw("μ$i", Data.SampledFrom([1, -1, 2, 1 // 2])) * f[i]' * f[i] for i in 1:n)
        H, Hbdg = hilbert_space(f, 1:n), hilbert_space(f, 1:n, BdG())
        vals, vecs = eigen(Hermitian(representation(ham, H, :dense)))
        assume!(vals[2] - vals[1] > 1.0e-6)
        F = eigen(representation(ham, Hbdg, :dense), BdGEigen())
        G = correlation_matrix(F)
        expectation(ham, G, Hbdg) ≊ vals[1] && G ≊ correlation_matrix(vecs[:, 1], H, Hbdg)
    end
end
