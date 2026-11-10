
"""
    @boson b

Create a symbolic bosonic annihilation operator `b`.
Use `b'` for the corresponding creation operator.
"""
macro boson(x)
    return Expr(
        :block, :($(esc(x)) = Field{Boson}($(Expr(:quote, x)))[]),
        :($(esc(x))))
end
"""
    @bosons b

Create a bosonic field `b`. Index into it to get bosonic mode operators:
`b[i]` returns the annihilation operator for mode `i`, `b[i]'` for the creation operator.
"""
macro bosons(name)
    return Expr(
        :block, :($(esc(name)) = Field{Boson}($(Expr(:quote, name)))),
        :($(esc(name))))
end


# `exp` is the power: `-k` for `b^k` and `k` for `b'^k`.
struct Boson
    exp::Int
end
const BosonSym{M} = Op{Boson, M}
mode_operator(::Type{Boson}) = Boson(-1)
Base.adjoint(k::Boson) = Boson(-k.exp)
function Base.show(io::IO, x::BosonSym)
    exp = x.kind.exp
    show(io, x.mode.field)
    print(io, exp > 0 ? "†" : "")
    if abs(exp) !== 1
        print(io, "^", abs(exp))
    end
    return _show_label(io, x.mode.label)
end
# Normal ordering of a = b^m, c = b†^n: b^m b†^n = Σₖ k! C(m,k) C(n,k) b†^(n-k) b^(m-k)
function _boson_normal_order(a::BosonSym, c::BosonSym)
    m, n = -a.kind.exp, c.kind.exp
    terms = map(0:min(m, n)) do k
        factors = typeof(a)[]
        n > k && push!(factors, Op(Boson(n - k), a.mode))
        m > k && push!(factors, Op(Boson(k - m), a.mode))
        # Widening to BigInt on overflow would make mul_effect's return type a union too wide to split,
        # which slows down every boson product, so overflow throws instead.
        NCMul(Base.Checked.checked_mul(factorial(k), binomial(m, k), binomial(n, k)), factors)
    end
    return AddTerms(terms)
end
function local_mul_effect(a::BosonSym, b::BosonSym)
    m, n = a.kind.exp, b.kind.exp
    sign(m) == sign(n) && return Op(Boson(m + n), a.mode)
    return m < 0 && n > 0 ? _boson_normal_order(a, b) : nothing
end

@testitem "BosonSym" begin
    using Symbolics
    @variables a::Real z::Complex

    @boson b1
    @boson b2
    @fermions f
    @majoranas γ

    @test 1 * b1 == b1
    @test 1 * b1 + 0 == b1
    @test 1 * b1 + 0 == 1 * b1
    @test hash(b1) == hash(1 * b1) == hash(1 * b1 + 0)

    # Test canonical commutation relations
    @test b1 * b1' - b1' * b1 == 1
    @test b1' * b1 - b1 * b1' == -1

    # Commutators involving powers: [b^m, b†] = m b^(m-1), [b, b†^n] = n b†^(n-1)
    for k in 1:4
        @test b1^k * b1' - b1' * b1^k == k * b1^(k - 1)
        @test b1 * b1'^k - b1'^k * b1 == k * b1'^(k - 1)
    end
    @test b1^2 * b1'^2 == b1'^2 * b1^2 + 4 * b1' * b1 + 2
    @test b1^3 * b1'^2 == b1'^2 * b1^3 + 6 * b1' * b1^2 + 6 * b1
    @test (b1 * b1') * b1 == b1 * (b1' * b1)
    @test (b1^2 * b1') * b1'^2 == b1^2 * (b1' * b1'^2)

    # Bosons at different sites commute
    @test b1 * b2 - b2 * b1 == 0
    @test b1' * b2 - b2 * b1' == 0
    @test b1' * b2' - b2' * b1' == 0

    # Bosons commute with fermions
    @test b1 * f[1] - f[1] * b1 == 0
    @test b1' * f[1] - f[1] * b1' == 0
    @test b1 * f[1]' - f[1]' * b1 == 0
    @test b1' * f[1]' - f[1]' * b1' == 0

    # Bosons commute with majoranas
    @test b1 * γ[1] - γ[1] * b1 == 0
    @test b1' * γ[1] - γ[1] * b1' == 0

    @test iszero(b1 - b1)
    @test iszero(2 * b1 - 2 * b1)
    @test iszero(0 * b1)
    @test iszero(b1 * 0)
    @test iszero(0 * (b1 + b2))
    @test iszero((b1 + b2) * 0)

    # Test adjoint
    @test b1'' == b1
    @test (b1 * b2)' == b2' * b1'

    # Test algebraic properties
    @test 1 + (b1 + b2) == 1 + b1 + b2 == b1 + b2 + 1 == b1 + 1 + b2
    @test (b1 * b2) * b1' == b1 * (b2 * b1')
end

"""
    BosonicState(n)

The basis state with `n` bosons in a single mode, as returned by `basisstates` of a bosonic space.
"""
struct BosonicState <: AbstractBasisState
    n::Int
end
Base.:(==)(a::BosonicState, b::BosonicState) = a.n == b.n
Base.isless(a::BosonicState, b::BosonicState) = a.n < b.n
Base.hash(x::BosonicState, h::UInt) = hash(x.n, h)
internal_rep(state::BosonicState, ::AbstractHilbertSpace, ::Type{T}) where T<:Integer = T(state.n)
physical_rep(state::T, ::Type{BosonicState}) where T<:Integer = BosonicState(Int(state))

struct TruncatedBosonicHilbertSpace{M} <: AbstractAtomicHilbertSpace{BosonicState}
    mode::M
    dimension::Int
    function TruncatedBosonicHilbertSpace(mode::M, dimension::Integer) where {M}
        if dimension < 1
            throw(ArgumentError("Bosonic Hilbert space dimension must be positive, got $dimension"))
        end
        return new{M}(mode, Int(dimension))
    end
end
Base.:(==)(a::TruncatedBosonicHilbertSpace, b::TruncatedBosonicHilbertSpace) = a === b || (a.mode == b.mode && a.dimension == b.dimension)
Base.hash(x::TruncatedBosonicHilbertSpace, h::UInt) = hash(x.mode, hash(x.dimension, h))
basisstates(H::TruncatedBosonicHilbertSpace) = BasisStates(H, Iterators.map(BosonicState, 0:(dim(H)-1)))
Base.@propagate_inbounds function basisstate(n::Integer, H::TruncatedBosonicHilbertSpace)
    @boundscheck 1 <= n <= dim(H) || throw(BoundsError(H, n))
    BosonicState(n - 1)
end
dim(H::TruncatedBosonicHilbertSpace) = H.dimension
_state_index(s::BosonicState, H::TruncatedBosonicHilbertSpace) = 0 <= s.n < dim(H) ? s.n + 1 : 0
has_arithmetic_state_index(::TruncatedBosonicHilbertSpace) = true
add_tag(H::TruncatedBosonicHilbertSpace, tag) = TruncatedBosonicHilbertSpace(add_tag(H.mode, tag), H.dimension)

"""
    hilbert_space(b::BosonSym, dimension[, constraint])
    hilbert_space(b::Field{Boson}, labels, dimension[, constraint])

The truncated space of the bosonic mode `b`, with basis states `|0⟩, …, |dimension-1⟩`, so the
maximal occupation is `dimension - 1`. With `labels`, the tensor product of one such space for
each mode `b[l]`, `l in labels`. An optional `constraint` restricts the basis states.

Operator terms that would raise the occupation above the cutoff are dropped without error,
while leaving a constrained space throws unless `projection=true`. Products are normal ordered
before truncation, so `b * b'` is represented as `b'b + 1`.
"""
hilbert_space(m::Mode{<:Field{Boson}}, dimension::Integer) = TruncatedBosonicHilbertSpace(m, dimension)
hilbert_space(m::Mode{<:Field{Boson}}, dimension::Integer, @nospecialize(constraint::ConstraintLike)) = constrain_space(hilbert_space(m, dimension), constraint)
particle_number(s::BosonicState) = s.n
parity(s::BosonicState) = iseven(s.n) ? 1 : -1

@inline function apply_local_operator(factor::Boson, state::BosonicState, space::TruncatedBosonicHilbertSpace, precomp, ::Type{T} = Float64) where {T}
    n = state.n
    k = abs(factor.exp)
    amplitude = one(T)
    if factor.exp < 0
        if n < k
            return state, zero(T)
        end
        for i in 0:(k-1)
            amplitude *= sqrt(T(n - i))
        end
        n -= k
    else
        for i in 1:k
            amplitude *= sqrt(T(n + i))
        end
        n += k
    end
    if n >= dim(space)
        return state, zero(T)
    end
    return BosonicState(n), amplitude
end
atomic_id(H::TruncatedBosonicHilbertSpace) = H.mode

@testitem "Bosonic hilbert space" begin
    using FermionicHilbertSpaces: TruncatedBosonicHilbertSpace, BosonicState, state_index, basisstate
    @boson b
    H = TruncatedBosonicHilbertSpace(b.mode, 4)
    @test collect(basisstates(H)) == [BosonicState(0), BosonicState(1), BosonicState(2), BosonicState(3)]
    @test dim(H) == 4
    @test basisstate(1, H) == BosonicState(0)
    @test basisstate(4, H) == BosonicState(3)
    @test state_index(BosonicState(0), H) == 1
    @test state_index(BosonicState(3), H) == 4
    # states outside the truncation are not in H
    @test state_index(BosonicState(4), H) == 0
    @test state_index(BosonicState(-1), H) == 0
end

@testitem "Bosonic matrix representations" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: TruncatedBosonicHilbertSpace

    @boson b
    H = TruncatedBosonicHilbertSpace(b.mode, 4)

    d = dim(H)
    a = spzeros(Float64, d, d)
    for n in 1:(d-1)
        a[n, n+1] = sqrt(n)
    end

    ma = representation(b, H)
    madag = representation(b', H)

    @test ma isa SparseMatrixCSC
    @test madag isa SparseMatrixCSC
    @test ma == a
    @test madag == a'
    @test representation(b, H) == ma
    @test madag * ma + 1im * I ≈ representation(b' * b + 1im, H)
    # the truncation breaks [a, a†] = 1 only at the cutoff
    @test ma * madag - madag * ma ≈ Diagonal([ones(d - 1); -(d - 1)])
end

@testitem "Bosonic triple products vs matrices" begin
    @boson b
    N = 10
    H = hilbert_space(b, N)
    ops = [b, b', b^2, b'^2]
    mats = [representation(op, H) for op in ops]
    low = 1:(N-2) # avoid truncation effects
    for i in eachindex(ops), j in eachindex(ops), k in eachindex(ops)
        sym = representation(ops[i] * ops[j] * ops[k], H)
        @test sym[low, low] ≈ (mats[i]*mats[j]*mats[k])[low, low]
    end
end

@testitem "Bosonic normal ordering of large powers" begin
    @boson b
    H = hilbert_space(b, 25)
    low = 1:5 # states that b'^18 keeps below the cutoff
    ref = representation(b, H, Matrix)^18 * representation(b', H, Matrix)^18
    @test representation(b^18 * b'^18, H, Matrix)[low, low] ≈ ref[low, low]
    @test_throws OverflowError b^19 * b'^19
end

@testitem "Boson product spaces and number conservation" begin
    using Combinatorics: binomial
    using LinearAlgebra: I, norm

    function constrained_boson_dim(nr_of_modes, max_occ, total_particles)
        L, nmax, M = nr_of_modes, max_occ, total_particles
        (M < 0 || M > L * nmax) && return 0
        s = 0
        for k in 0:fld(M, nmax+1)
            s += (-1)^k * binomial(L, k) * binomial(M - k * (nmax + 1) + L - 1, L - 1)
        end
        return s
    end

    N = 6
    total_particles = N
    local_dimension = 4
    max_occupancy = local_dimension - 1
    @bosons b

    Hfull = hilbert_space(b, 1:N, local_dimension)
    @test dim(Hfull) == local_dimension^N
    H = constrain_space(Hfull, NumberConservation(total_particles))
    @test dim(H) == constrained_boson_dim(N, max_occupancy, total_particles)
    Nop = sum(b[i]' * b[i] for i in 1:N)
    @test norm(representation(Nop, H) - total_particles * I(dim(H))) < 1e-10
end

@testitem "Fermion-spin-boson mixed spaces" begin
    using LinearAlgebra

    @fermions f
    @spin S
    @boson b

    Hf = hilbert_space(f, 1:1)
    Hs = hilbert_space(S, 1 // 2)
    Hb = hilbert_space(b, 3)
    H = tensor_product(Hf, Hs, Hb)

    f_expr = f[1]' * f[1] + 0.5 * (f[1] + f[1]') + 1
    s_expr = S[:z] + 0.5 * (S[:x] + S[:y]) + 1
    b_expr = b' * b + 0.5 * (b + b') + 1
    fmat = representation(f_expr, Hf)
    smat = representation(s_expr, Hs)
    bmat = representation(b_expr, Hb)
    op = f_expr * s_expr * b_expr
    mop = representation(op, H)
    expected = kron(reverse([fmat, smat, bmat])...)
    @test mop ≈ expected

    @test_throws ArgumentError representation(s_expr, hilbert_space((@spin Not), 1 // 2))

    Hsb = tensor_product(Hs, Hb)
    @test partial_trace(1.0 * I(dim(Hsb)), Hsb => Hs) ≈ dim(Hb) * I(dim(Hs))
    @test partial_trace(1.0 * I(dim(Hsb)), Hsb => Hb) ≈ dim(Hs) * I(dim(Hb))

    trf = tr(fmat)
    trs = tr(smat)
    trb = tr(bmat)
    @test partial_trace(mop, H => Hf; complement=Hsb) ≈ trb * trs * fmat
    @test partial_trace(mop, H => Hf) ≈ trb * trs * fmat
    @test partial_trace(mop, H => Hs) ≈ trb * trf * smat
    @test partial_trace(mop, H => Hb) ≈ trs * trf * bmat

    mf = rand(dim(Hf), dim(Hf))
    ms = rand(dim(Hs), dim(Hs))
    mb = rand(dim(Hb), dim(Hb))
    @test embed(mf, Hf => H) ≈ kron(I(dim(Hs)), I(dim(Hb)), mf)

    @test partial_trace(embed(mf, Hf => H), H => Hf) ≈ (dim(Hs) * dim(Hb)) * mf
    @test partial_trace(embed(ms, Hs => H), H => Hs) ≈ (dim(Hf) * dim(Hb)) * ms
    @test partial_trace(embed(mb, Hb => H), H => Hb) ≈ (dim(Hf) * dim(Hs)) * mb

    # subregions
    @test Hf == subregion([f[1]], H)
    @test Hs == subregion([S], H)
    @test Hb == subregion([b], H)
end

@testitem "Reduced states are physical" begin
    using LinearAlgebra

    @fermions f
    @spin S
    @boson b
    Hf = hilbert_space(f, 1:1)
    Hs = hilbert_space(S, 1 // 2)
    Hb = hilbert_space(b, 3)
    H = tensor_product(Hf, Hs, Hb)
    ψ = normalize(rand(ComplexF64, dim(H)))
    ρ = ψ * ψ'
    for Hsub in (Hf, Hs, Hb)
        ρsub = partial_trace(ρ, H => Hsub)
        @test ρsub ≈ ρsub'
        @test tr(ρsub) ≈ 1
        @test eigmin(Hermitian(ρsub)) >= -1e-12
    end

    # the singlet (|↑↓⟩ - |↓↑⟩)/√2 is maximally entangled
    @spin S1 1 // 2
    @spin S2 1 // 2
    H1 = hilbert_space(S1)
    H12 = tensor_product(H1, hilbert_space(S2))
    singlet = [0, 1, -1, 0] / sqrt(2)
    @test partial_trace(singlet * singlet', H12 => H1) ≈ I / 2
end

