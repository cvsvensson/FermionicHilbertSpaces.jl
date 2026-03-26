
"""
    @boson b

Create a symbolic bosonic annihilation operator `b`.
Use `b'` for the corresponding creation operator.
"""
macro boson(x)
    Expr(:block, :($(esc(x)) = BosonSym($(Expr(:quote, x)), -1)),
        :($(esc(x))))
end
macro bosons(name, labels)
    Expr(:block,
        :($(esc(name)) = OrderedDict(l => BosonSym(Symbol($(Expr(:quote, name)), l), -1) for l in $(esc(labels)))),
        :($(esc(name))))
end

struct BosonSym{B} <: AbstractSym
    name::B
    exp::Int
end
Base.adjoint(x::BosonSym) = BosonSym(x.name, -x.exp)
Base.iszero(x::BosonSym) = false
function Base.show(io::IO, x::BosonSym)
    print(io, x.name, x.exp > 0 ? "†" : "")
    if abs(x.exp) !== 1
        print(io, "^", abs(x.exp))
    end
end
Base.:(==)(a::BosonSym, b::BosonSym) = a.exp == b.exp && a.name == b.name
Base.hash(a::BosonSym, h::UInt) = hash(a.exp, hash(a.name, h))

function NonCommutativeProducts.mul_effect(a::BosonSym, b::BosonSym)
    if a.name == b.name
        if sign(a.exp) == sign(b.exp)
            return BosonSym(a.name, a.exp + b.exp)
        else
            if a.exp < 0 && b.exp > 0
                return AddTerms((Swap(1), 1))
            else
                return nothing
            end
        end
    end
    if a.name > b.name
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

struct BosonicFockState <: AbstractBasisState
    n::Int
end
Base.:(==)(a::BosonicFockState, b::BosonicFockState) = a.n == b.n
Base.hash(x::BosonicFockState, h::UInt) = hash(x.n, h)
struct TruncatedBosonicHilbertSpace{B} <: AbstractAtomicHilbertSpace{BosonicFockState}
    sym::BosonSym{B}
    max_occupancy::Int
    function TruncatedBosonicHilbertSpace(sym::BosonSym{B}, max_occupancy) where B
        new{B}(sym, max_occupancy)
    end
end
Base.:(==)(a::TruncatedBosonicHilbertSpace, b::TruncatedBosonicHilbertSpace) = a === b || (a.sym == b.sym && a.max_occupancy == b.max_occupancy)
Base.hash(x::TruncatedBosonicHilbertSpace, h::UInt) = hash(x.sym, hash(x.max_occupancy, h))
basisstates(H::TruncatedBosonicHilbertSpace) = map(BosonicFockState, 0:H.max_occupancy)
function basisstate(n::Int, H::TruncatedBosonicHilbertSpace)
    if n < 1 || n > H.max_occupancy + 1
        throw(ArgumentError("Basis state index $n is out of bounds for Hilbert space with max occupancy $(H.max_occupancy)"))
    end
    BosonicFockState(n - 1)
end
dim(H::TruncatedBosonicHilbertSpace) = H.max_occupancy + 1
function Base.show(io::IO, H::TruncatedBosonicHilbertSpace)
    if get(io, :compact, false)
        print(io, "Bosons(", H.sym.name, ", max=", H.max_occupancy, ")")
    else
        print(io, "$(dim(H))-dimensional TruncatedBosonicHilbertSpace\n")
        print(io, "Label: ", H.sym.name, ", max_occupancy: ", H.max_occupancy)
    end
end
function state_index(s::BosonicFockState, H::TruncatedBosonicHilbertSpace)
    if s.n < 0 || s.n > H.max_occupancy
        throw(ArgumentError("State $s is not in the Hilbert space"))
    end
    s.n + 1
end

hilbert_space(sym::BosonSym{B}, max_occupancy) where B = TruncatedBosonicHilbertSpace(sym, max_occupancy)

function apply_local_operators(op::NCMul, state::BosonicFockState, space::TruncatedBosonicHilbertSpace, precomp)
    factors = op.factors
    n = state.n
    amplitude = op.coeff

    for factor in Iterators.reverse(factors)
        k = abs(factor.exp)
        if factor.exp < 0
            if n < k
                return ((state, 0 * amplitude),)
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
    if n > space.max_occupancy
        return ((state, 0 * amplitude),)
    end
    return ((BosonicFockState(n), amplitude),)
end

symbolic_group(f::BosonSym) = BosonSym(f.name, 0)
symbolic_group(H::TruncatedBosonicHilbertSpace) = symbolic_group(H.sym)
mat_eltype(::Type{S}) where {S<:BosonSym} = Float64

@testitem "Bosonic hilbert space" begin
    using FermionicHilbertSpaces: TruncatedBosonicHilbertSpace, BosonicFockState, state_index, basisstate
    @boson b
    H = TruncatedBosonicHilbertSpace(b, 3)
    @test basisstates(H) == [BosonicFockState(0), BosonicFockState(1), BosonicFockState(2), BosonicFockState(3)]
    @test dim(H) == 4
    @test basisstate(1, H) == BosonicFockState(0)
    @test basisstate(4, H) == BosonicFockState(3)
    @test state_index(BosonicFockState(0), H) == 1
    @test state_index(BosonicFockState(3), H) == 4
end

@testitem "Bosonic matrix representations" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: TruncatedBosonicHilbertSpace

    @boson b
    H = TruncatedBosonicHilbertSpace(b, 3)

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

@testitem "Fermion-spin-boson mixed spaces" begin
    using LinearAlgebra

    @fermions f
    @spin S
    @boson b

    Hf = hilbert_space(f, 1:1)
    Hs = hilbert_space(S, 1 // 2)
    Hb = hilbert_space(b, 2)
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
end

