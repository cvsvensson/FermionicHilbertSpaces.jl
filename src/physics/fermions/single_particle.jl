const SingleParticleState = FixedNumberFockState{1}
SingleParticleState(site::Int) = FixedNumberFockState((site,))

struct SingleParticleHilbertSpace{H}
    parent::H
    function SingleParticleHilbertSpace(f::SymbolicFermionBasis, labels)
        states = [SingleParticleState(i) for (i, label) in enumerate(labels)]
        H = hilbert_space(f, labels, states)
        return new{typeof(H)}(H)
    end
end
dim(h::SingleParticleHilbertSpace) = dim(h.parent)
Base.parent(h::SingleParticleHilbertSpace) = h.parent
Base.keys(h::SingleParticleHilbertSpace) = keys(h.parent)
mode_ordering(h::SingleParticleHilbertSpace) = mode_ordering(h.parent)
modes(h::SingleParticleHilbertSpace) = modes(h.parent)
basisstates(h::SingleParticleHilbertSpace) = basisstates(h.parent)
Base.:(==)(a::SingleParticleHilbertSpace, b::SingleParticleHilbertSpace) = a === b || a.parent == b.parent
Base.hash(x::SingleParticleHilbertSpace, h::UInt) = hash(x.parent, h)
function Base.show(io::IO, H::SingleParticleHilbertSpace)
    return if get(io, :compact, false)
        print(io, "SingleParticleHilbertSpace($(dim(H))-dim)")
    else
        print(io, "$(dim(H))-dimensional SingleParticleHilbertSpace\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), parent(H))
    end
end
"""
    SingleParticle()

Marker passed as the last argument of `hilbert_space(f, labels, SingleParticle())` to select the single-particle space of non-interacting fermions with number conservation.
"""
struct SingleParticle end

"""
    hilbert_space(f::SymbolicFermionBasis, labels, ::SingleParticle)

The space spanned by the states ``c_n^\\dagger |0\\rangle`` for the modes `f[l]`, `l in labels`, suitable for non-interacting systems with fermion number conservation. The matrix representation of a quadratic operator is the single-particle matrix, without any contribution from the identity operator. The space wraps a constrained fermionic space, available through `parent`, and does not combine with tensor products or partial traces.
"""
hilbert_space(f::SymbolicFermionBasis, labels, ::SingleParticle) = SingleParticleHilbertSpace(f, labels)
Base.@propagate_inbounds basisstate(ind::Integer, H::SingleParticleHilbertSpace) = basisstate(ind, parent(H))
_state_index(state, H::SingleParticleHilbertSpace) = _state_index(state, parent(H))

@testitem "Single particle hilbert space" begin
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra
    @fermions f
    H = hilbert_space(f, 1:2, SingleParticle())
    opmul = f[1]' * f[2]
    @test representation(opmul, H) ≈ representation(opmul, parent(H))
    opadd = opmul + hc
    ham = representation(opadd, H)
    @test ham ≈ representation(opadd, parent(H))
    @test representation(opadd + I, H) == representation(opadd, H)
    @test representation(opadd + I, H) == representation(opadd + I, parent(H)) - I
end

@testitem "Single particle space representation formats" begin
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    H = hilbert_space(f, 1:3, SingleParticle())
    op = sum(rand(ComplexF64) * f[i]' * f[j] for i in 1:3, j in 1:3)
    M = representation(op, H)
    D = representation(op, H, :dense)
    @test D isa Matrix && D == M
    @test representation(op, H, ComplexF64) == M
    @test_throws ArgumentError representation(op, H, :lazy)
end

@testitem "Single particle expectation values" begin
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra
    @fermions f
    N = 3
    H = hilbert_space(f, 1:N, SingleParticle())
    Hfock = hilbert_space(f, 1:N, NumberConservation())
    ham = sum(rand(ComplexF64) * f[i]' * f[j] + hc for i in 1:N, j in 1:N)
    ρ = exp(-representation(ham, Hfock, :dense))
    ρ /= tr(ρ)
    G = [tr(ρ * representation(f[i]' * f[j], Hfock, :dense)) for i in 1:N, j in 1:N]
    op = f[1]' * f[2] + 2 * f[3] * f[1]' + 1
    @test expectation(op, G, H) ≈ tr(ρ * representation(op, Hfock, :dense))
end

representation(op, H::SingleParticleHilbertSpace, repr = EagerSparseRepr(); chunking = NoChunking(), kwargs...) = matrix_representation(op, H, repr; chunking = chunking, kwargs...)
function matrix_representation(op, H::SingleParticleHilbertSpace, repr = EagerSparseRepr(); chunking = NoChunking(), kwargs...)
    _isscalar_operator(op) && return _identity_free_zero(H, repr)
    _is_quadratic_fermion_operator(op, true) || throw(ArgumentError("Only quadratic, number conserving fermionic operators can be represented on a SingleParticleHilbertSpace. Use `hilbert_space(f, labels)` and `representation(op, H)` for other operators."))
    return _quadratic_matrix(_single_particle_entry, remove_identity(op), H, repr)
end
function _single_particle_entry(term, H)
    a, b = term.factors
    outind = state_index(SingleParticleState(mode_position(a, H)), H)
    inind = state_index(SingleParticleState(mode_position(b, H)), H)
    return outind, inind, (-1)^b.creation * term.coeff
end
