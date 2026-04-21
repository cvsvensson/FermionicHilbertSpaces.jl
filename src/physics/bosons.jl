
"""
    @boson b

Create a symbolic bosonic annihilation operator `b`.
Use `b'` for the corresponding creation operator.
"""
macro boson(x)
    Expr(:block, :($(esc(x)) = BosonSym($(Expr(:quote, x)), nothing, -1)),
        :($(esc(x))))
end
"""
    @bosons b

Create a bosonic basis `b`. Index into it to get bosonic mode operators:
`b[i]` returns the annihilation operator for mode `i`, `b[i]'` for the creation operator.
"""
macro bosons(name)
    Expr(:block, :($(esc(name)) = BosonField($(Expr(:quote, name)))),
        :($(esc(name))))
end

struct BosonField{T<:Tags}
    name::Symbol
    tags::T
end
BosonField(name::Symbol) = BosonField(name, Tags(nothing))
Base.:(==)(a::BosonField, b::BosonField) = a.name == b.name && a.tags == b.tags
Base.hash(b::BosonField, h::UInt) = hash(b.tags, hash(b.name, h))
Base.show(io::IO, b::BosonField) = print(io, "BosonField(", b.name, ")")
tags(b::BosonField) = b.tags
add_tag(b::BosonField, tag) = BosonField(b.name, add_tag(b.tags, tag))

struct BosonSym{L,B} <: AbstractSym
    label::L
    basis::B
    exp::Int
end
Base.getindex(b::BosonField, i) = BosonSym(i, b, -1)
Base.adjoint(x::BosonSym) = BosonSym(x.label, x.basis, -x.exp)
Base.iszero(x::BosonSym) = false
symbolic_basis(x::BosonSym{<:Any,<:BosonField}) = x.basis
change_basis(x::BosonSym, newbasis) = BosonSym(x.label, newbasis, x.exp)
function Base.show(io::IO, x::BosonSym)
    if x.basis isa Nothing
        print(io, x.label)
    else
        print(io, _symbolic_name_with_tags(x.basis.name, x.basis))
    end
    print(io, x.exp > 0 ? "†" : "")
    if abs(x.exp) !== 1
        print(io, "^", abs(x.exp))
    end
    if !(x.basis isa Nothing)
        print(io, "[", x.label, "]")
    end
end
Base.:(==)(a::BosonSym, b::BosonSym) = a.exp == b.exp && a.label == b.label && isequal(a.basis, b.basis)
Base.hash(a::BosonSym, h::UInt) = hash(a.exp, hash(a.label, hash(a.basis, h)))
_boson_name(s::BosonSym) = s.basis isa Nothing ? (s.label) : Symbol(s.basis.name, s.label)
function NonCommutativeProducts.mul_effect(a::BosonSym, b::BosonSym)
    if _boson_name(a) == _boson_name(b)
        if sign(a.exp) == sign(b.exp)
            return BosonSym(a.label, a.basis, a.exp + b.exp)
        else
            if a.exp < 0 && b.exp > 0
                return AddTerms((Swap(1), 1))
            else
                return nothing
            end
        end
    end
    if _boson_name(a) > _boson_name(b)
        return Swap(1)
    else
        return nothing
    end
end

@nc BosonSym
NonCommutativeProducts.@commutative FermionSym BosonSym
NonCommutativeProducts.@commutative MajoranaSym BosonSym

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

struct BosonicState <: AbstractBasisState
    n::Int
end
Base.:(==)(a::BosonicState, b::BosonicState) = a.n == b.n
Base.isless(a::BosonicState, b::BosonicState) = a.n < b.n
Base.hash(x::BosonicState, h::UInt) = hash(x.n, h)
struct TruncatedBosonicHilbertSpace{L,B} <: AbstractAtomicHilbertSpace{BosonicState}
    sym::BosonSym{L,B}
    dimension::Int
    function TruncatedBosonicHilbertSpace(sym::BosonSym{L,B}, dimension::Integer) where {L,B}
        if dimension < 1
            throw(ArgumentError("Bosonic Hilbert space dimension must be positive, got $dimension"))
        end
        new{L,B}(sym, Int(dimension))
    end
end
Base.:(==)(a::TruncatedBosonicHilbertSpace, b::TruncatedBosonicHilbertSpace) = a === b || (a.sym == b.sym && a.dimension == b.dimension)
Base.hash(x::TruncatedBosonicHilbertSpace, h::UInt) = hash(x.sym, hash(x.dimension, h))
basisstates(H::TruncatedBosonicHilbertSpace) = map(BosonicState, 0:(dim(H)-1))
function basisstate(n::Int, H::TruncatedBosonicHilbertSpace)
    if n < 1 || n > dim(H)
        throw(ArgumentError("Basis state index $n is out of bounds for Hilbert space with dimension $(dim(H))"))
    end
    BosonicState(n - 1)
end
dim(H::TruncatedBosonicHilbertSpace) = H.dimension
function state_index(s::BosonicState, H::TruncatedBosonicHilbertSpace)
    if s.n < 0 || s.n >= dim(H)
        throw(ArgumentError("State $s is not in the Hilbert space"))
    end
    s.n + 1
end

hilbert_space(sym::BosonField, labels, dimension::Int, constraint::AbstractConstraint=NoSymmetry()) = tensor_product(map(l -> hilbert_space(sym[l], dimension), labels); constraint)
hilbert_space(sym::BosonSym, dimension::Int) = TruncatedBosonicHilbertSpace(sym, dimension)
hilbert_space(sym::BosonSym, dimension::Int, constraint::AbstractConstraint) = constrain_space(hilbert_space(sym, dimension), constraint)
particle_number(s::BosonicState) = s.n
parity(s::BosonicState) = iseven(s.n) ? 1 : -1
maximum_particles(H::TruncatedBosonicHilbertSpace) = dim(H) - 1

function apply_local_operators(op::NCMul, state::BosonicState, space::TruncatedBosonicHilbertSpace, precomp)
    factors = op.factors
    n = state.n
    amplitude = op.coeff

    for factor in Iterators.reverse(factors)
        k = abs(factor.exp)
        if factor.exp < 0
            if n < k
                return (state,), (zero(amplitude),)
            end
            for i in 0:(k-1)
                amplitude *= sqrt(n - i)
            end
            n -= k
        else
            for i in 1:k
                amplitude *= sqrt(n + i)
            end
            n += k
        end
    end
    if n >= dim(space)
        return (state,), (zero(amplitude),)
    end
    return (BosonicState(n),), (amplitude,)
end

symbolic_group(f::BosonSym{<:Any,B}) where B<:BosonField = (symbolic_group(f.basis), f.label)
symbolic_group(f::BosonSym{<:Any,Nothing}) = (BosonSym, f.label)
atomic_id(f::BosonSym) = symbolic_group(f)
symbolic_group(H::TruncatedBosonicHilbertSpace) = symbolic_group(H.sym)
atomic_id(H::TruncatedBosonicHilbertSpace) = atomic_id(H.sym)
group_id(H::TruncatedBosonicHilbertSpace) = atomic_id(H)
mat_eltype(::Type{S}) where {S<:BosonSym} = Float64

@testitem "Bosonic hilbert space" begin
    using FermionicHilbertSpaces: TruncatedBosonicHilbertSpace, BosonicState, state_index, basisstate
    @boson b
    H = TruncatedBosonicHilbertSpace(b, 4)
    @test basisstates(H) == [BosonicState(0), BosonicState(1), BosonicState(2), BosonicState(3)]
    @test dim(H) == 4
    @test basisstate(1, H) == BosonicState(0)
    @test basisstate(4, H) == BosonicState(3)
    @test state_index(BosonicState(0), H) == 1
    @test state_index(BosonicState(3), H) == 4
end

@testitem "Bosonic matrix representations" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: TruncatedBosonicHilbertSpace

    @boson b
    H = TruncatedBosonicHilbertSpace(b, 4)

    d = dim(H)
    a = spzeros(Float64, d, d)
    for n in 1:(d-1)
        a[n, n+1] = sqrt(n)
    end

    ma = matrix_representation(b, H)
    madag = matrix_representation(b', H)

    @test ma isa SparseMatrixCSC
    @test madag isa SparseMatrixCSC
    @test ma == a
    @test madag == a'
    @test matrix_representation(b, H) == ma
    @test madag * ma + 1im * I ≈ matrix_representation(b' * b + 1im, H)
end

@testitem "Boson product spaces and number conservation" begin
    using Combinatorics: binomial
    using LinearAlgebra: I, norm

    function constrained_boson_dim(nr_of_modes, max_occ, total_particles)
        L, nmax, M = nr_of_modes, max_occ, total_particles
        (M < 0 || M > L * nmax) && return 0
        s = 0
        for k in 0:fld(M, nmax + 1)
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
    @test norm(matrix_representation(Nop, H) - total_particles * I(dim(H))) < 1e-10
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
    fmat = matrix_representation(f_expr, Hf)
    smat = matrix_representation(s_expr, Hs)
    bmat = matrix_representation(b_expr, Hb)
    op = f_expr * s_expr * b_expr
    mop = matrix_representation(op, H)
    expected = kron(reverse([fmat, smat, bmat])...)
    @test mop ≈ expected

    @test_throws ArgumentError matrix_representation(s_expr, FermionicHilbertSpaces.SpinSpace{1 // 2}(:Not))

    Hsb = tensor_product(Hs, Hb)
    @test partial_trace(1.0 * I(dim(Hsb)), Hsb => Hs) ≈ dim(Hb) * I(dim(Hs))
    @test partial_trace(1.0 * I(dim(Hsb)), Hsb => Hb) ≈ dim(Hs) * I(dim(Hb))

    trf = tr(fmat)
    trs = tr(smat)
    trb = tr(bmat)
    partial_trace(mop, H => Hf; complement=Hsb) ≈ trb * trs * fmat
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

