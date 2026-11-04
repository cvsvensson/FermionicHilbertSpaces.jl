"""
    FreeFermions

Hilbert spaces and tools for non-interacting fermions: the single-particle space
`hilbert_space(f, labels, SingleParticle())`, the Bogoliubov-de Gennes space
`hilbert_space(f, labels, BdG())`, the Majorana space `hilbert_space(γ, labels, SingleParticle())`,
their eigendecomposition and Gaussian states.
"""
module FreeFermions

    using ..FermionicHilbertSpaces: FermionicHilbertSpaces, AbstractFockState, AbstractHilbertSpace,
        EagerDenseRepr, EagerSparseRepr, FermionSym, FixedNumberFockState, MajoranaSym, NoChunking, SymbolicFermionBasis,
        SymbolicMajoranaBasis,
        _FermionModeSpace, _canonicalize_repr, _operator_type, _zero_matrix, exchange_phase, finalize!,
        matrix_accumulator, mode_position, push_inds_amps!, state_index
    import ..FermionicHilbertSpaces: _precomputation_before_operator_application, _state_index,
        _state_label_atomic, _unwrap_space, apply_local_operators, atom_ids, basisstate, basisstates, dim, von_neumann_entropy, expectation,
        group_id, hilbert_space, interpret_state, matrix_representation, mode_ordering, modes, partial_trace, representation,
        statetype, vector_representation
    using LinearAlgebra: LinearAlgebra, Diagonal, Eigen, Hermitian, Symmetric, UniformScaling, I,
        det, eigen, issymmetric, normalize, svd
    using NonCommutativeProducts: NonCommutativeProducts, NCAdd, NCMul, NCterms
    using OrderedCollections: OrderedDict

    using TestItems: @testitem

    export SingleParticle, BdG, BdGEigen, NambuState, SingleParticleState, quasiparticle, symbolic_operator,
        correlation_matrix, ground_state_parity, gaussian_state, mean_field

    remove_identity(a::NCMul) = a
    remove_identity(a::NCAdd) = NCAdd(zero(a.coeff), a.dict)

    # The single-particle and BdG spaces represent an operator without its identity part, so a scalar is represented by zero
    _isscalar_operator(op) = op isa Union{Number, UniformScaling} || (op isa Union{NCMul, NCAdd} && NonCommutativeProducts.isscalar(op))
    _identity_free_zero(H, repr) = _zero_matrix(H, _eager_repr(repr, H), Int)
    function _eager_repr(repr, H)
        _repr = _canonicalize_repr(repr)
        _repr isa Union{EagerSparseRepr, EagerDenseRepr} || throw(ArgumentError("Only the :sparse and :dense representations are supported on $(typeof(H)). Got $repr."))
        return _repr
    end
    # The matrix of a sum of two-fermion terms, where `entry(term, H)` gives the row, column and value of the one entry of each term
    function _quadratic_matrix(entry, op, H, repr)
        accumulator = matrix_accumulator(op, H, _eager_repr(repr, H))
        for term in NCterms(op)
            outind, inind, amp = entry(term, H)
            (iszero(outind) || iszero(inind)) && throw(ArgumentError("The operator `$term` contains a factor that is not part of the fermionic space `$H`."))
            push_inds_amps!(accumulator, outind, inind, amp)
        end
        return finalize!(accumulator, H)
    end

    # Whether every term of `op` is a product of two factors `a` and `b` with `isvalid(a, b)`
    _is_quadratic_operator(op, isvalid) = false
    _is_quadratic_operator(op::NCMul, isvalid) = length(op.factors) == 2 && isvalid(op.factors...)
    _is_quadratic_operator(op::NCAdd, isvalid) = all(term -> _is_quadratic_operator(term, isvalid), keys(op.dict))
    # Whether every term of `op` is a product of two `FermionSym`s, with one creation and one annihilation
    # operator if `number_conserving`
    _is_quadratic_fermion_operator(op, number_conserving) = _is_quadratic_operator(op, (a, b) -> a isa FermionSym && b isa FermionSym && (!number_conserving || a.creation != b.creation))

    function _mode_positions(H, Hsub)
        positions = [mode_position(m, H) for m in modes(Hsub)]
        any(iszero, positions) && throw(ArgumentError("The modes of $Hsub are not a subset of the modes of $H."))
        return positions
    end

    include("single_particle.jl")
    include("free_majoranas.jl")
    include("bdg.jl")
    include("gaussian.jl")
    include("wick.jl")

    # These spaces have the basis of their parent, where their kets and bras are represented
    const _ParentBasisSpace = Union{SingleParticleHilbertSpace, BdGHilbertSpace}
    statetype(H::_ParentBasisSpace) = statetype(parent(H))
    interpret_state(state, H::_ParentBasisSpace) = interpret_state(state, parent(H))
    group_id(H::_ParentBasisSpace) = group_id(parent(H))
    atom_ids(H::_ParentBasisSpace) = atom_ids(parent(H))
    _unwrap_space(H::_ParentBasisSpace) = _unwrap_space(parent(H))
    vector_representation(op, H::_ParentBasisSpace, repr = EagerSparseRepr(); kwargs...) = vector_representation(op, parent(H), repr; kwargs...)
    function representation(op, H::_ParentBasisSpace, repr = EagerSparseRepr(); chunking = NoChunking(), kwargs...)
        _operator_type(op) in (:kets, :bras) && return representation(op, parent(H), repr; kwargs...)
        return matrix_representation(op, H, repr; chunking, kwargs...)
    end

end

@testitem "Kets of single-particle and BdG spaces" begin
    using FermionicHilbertSpaces: vector_representation
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra
    @fermions f
    for H in (hilbert_space(f, 1:3, SingleParticle()), hilbert_space(f, 1:3, BdG()))
        basis = Matrix(I, dim(H), dim(H))
        kets = Kets(H)
        @test all(i -> representation(Ket(basisstate(i, H), H), H, :dense) == basis[:, i], 1:dim(H))
        @test representation(kets[1] + 2 * kets[2], H, :dense) == basis[:, 1] + 2 * basis[:, 2]
        @test vector_representation(kets[2]', H, :dense) == transpose(basis[:, 2])
        @test kets[2]' * (kets[1] + 3 * kets[2]) == 3
    end
    Hsp = hilbert_space(f, 1:3, SingleParticle())
    @test Ket("010", Hsp) == Kets(Hsp)[2]
    @test repr(Ket("010", Hsp)) == "|010⟩"
end
