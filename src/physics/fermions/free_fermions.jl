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
    _FermionModeSpace, _canonicalize_repr, _matrix_format, _repr_eltype, _xlogx, _zero_matrix, exchange_phase, finalize!,
        matrix_accumulator, mode_position, push_inds_amps!, state_index, SymbolicState
    import ..FermionicHilbertSpaces: _precomputation_before_operator_application, _state_index,
        _state_label_atomic, apply_local_operators, basisstate, basisstates, dim, von_neumann_entropy, expectation,
        hilbert_space, matrix_representation, mode_ordering, modes, partial_trace, representation, tensor_product
    using LinearAlgebra: LinearAlgebra, Diagonal, Eigen, Hermitian, Symmetric, UniformScaling, I,
    det, eigen, eigvals, issymmetric, kron, normalize, svd
    using SparseArrays: sparse, spzeros
    using NonCommutativeProducts: NonCommutativeProducts, NCAdd, NCMul, NCterms
    using OrderedCollections: OrderedDict

    using TestItems: @testitem

    export SingleParticle, BdG, BdGEigen, NambuState, SingleParticleState, quasiparticle, symbolic_operator,
        correlation_matrix, ground_state_parity, gaussian_state, mean_field

    remove_identity(a::NCMul) = a
    remove_identity(a::NCAdd) = NCAdd(zero(a.coeff), a.dict)

    # The single-particle and BdG spaces represent an operator without its identity part, so a scalar is represented by zero
    _isscalar_operator(op) = op isa Union{Number, UniformScaling} || (op isa Union{NCMul, NCAdd} && NonCommutativeProducts.isscalar(op))
function _identity_free_zero(H, repr)
    _repr = _eager_repr(repr, H)
    return _zero_matrix(H, _repr, _repr_eltype(Int, _repr))
end
    function _eager_repr(repr, H)
    _repr = _matrix_format(repr)
        _repr isa Union{EagerSparseRepr, EagerDenseRepr} || throw(ArgumentError("Only the :sparse and :dense representations are supported on $(typeof(H)). Got $repr."))
        return _repr
    end
# The matrix of a sum of two-fermion terms, where `entries(term, H)` gives a tuple of the (row, column, value) entries of each term
function _quadratic_matrix(entries, op, H, repr)
        accumulator = matrix_accumulator(op, H, _eager_repr(repr, H))
    for term in NCterms(op), (outind, inind, amp) in entries(term, H)
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

_check_size(G, H) = size(G) == (dim(H), dim(H)) ||
    throw(DimensionMismatch("The correlation matrix has size $(size(G)), but the space has dimension $(dim(H))."))

    function _mode_positions(H, Hsub)
        positions = [mode_position(m, H) for m in modes(Hsub)]
        any(iszero, positions) && throw(ArgumentError("The modes of $Hsub are not a subset of the modes of $H."))
        return positions
    end

    include("single_particle.jl")
    include("free_majoranas.jl")
    include("bdg.jl")
    include("free_product_space.jl")
    include("gaussian.jl")
    include("wick.jl")

end
