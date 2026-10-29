"""
    FreeFermions

Hilbert spaces and tools for non-interacting fermions: the single-particle space
`hilbert_space(f, labels, SingleParticle())`, the Bogoliubov-de Gennes space
`hilbert_space(f, labels, BdG())`, their eigendecomposition and Gaussian states.
"""
module FreeFermions

    using ..FermionicHilbertSpaces: FermionicHilbertSpaces, AbstractFockState, AbstractHilbertSpace,
        EagerDenseRepr, EagerSparseRepr, FermionSym, FixedNumberFockState, NoChunking, SymbolicFermionBasis,
        _FermionModeSpace, _canonicalize_repr, _zero_matrix, exchange_phase, finalize!,
        matrix_accumulator, mode_position, push_inds_amps!, state_index
    import ..FermionicHilbertSpaces: _precomputation_before_operator_application, _state_index,
        _state_label_atomic, apply_local_operators, basisstate, basisstates, dim, expectation,
        hilbert_space, matrix_representation, mode_ordering, modes, representation
    using LinearAlgebra: LinearAlgebra, Diagonal, Eigen, Hermitian, Symmetric, UniformScaling, I,
        det, eigen, issymmetric, normalize, svd
    using NonCommutativeProducts: NonCommutativeProducts, @nc, AddTerms, NCAdd, NCMul, NCterms, Swap

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

    # Whether every term of `op` is a product of two `FermionSym`s, with one creation and one annihilation
    # operator if `number_conserving`. Majoranas and other operators give `false`.
    _is_quadratic_fermion_operator(op, number_conserving) = false
    _is_quadratic_fermion_operator(op::NCMul, number_conserving) = _is_quadratic_fermion_term(op.factors, number_conserving)
    _is_quadratic_fermion_operator(op::NCAdd, number_conserving) = all(term -> _is_quadratic_fermion_term(term.factors, number_conserving), keys(op.dict))
    function _is_quadratic_fermion_term(factors, number_conserving)
        length(factors) == 2 || return false
        a, b = factors
        (a isa FermionSym && b isa FermionSym) || return false
        return !number_conserving || a.creation != b.creation
    end

    include("single_particle.jl")
    include("bdg.jl")
    include("gaussian.jl")
    include("wick.jl")

end

@deprecate single_particle_hilbert_space(f, labels) hilbert_space(f, labels, FermionicHilbertSpaces.FreeFermions.SingleParticle())
@deprecate bdg_hilbert_space(f, labels) hilbert_space(f, labels, FermionicHilbertSpaces.FreeFermions.BdG())

@testitem "Deprecated special space constructors" begin
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    Hsp = @test_deprecated single_particle_hilbert_space(f, 1:2)
    Hbdg = @test_deprecated bdg_hilbert_space(f, 1:2)
    @test Hsp == hilbert_space(f, 1:2, SingleParticle())
    @test Hbdg == hilbert_space(f, 1:2, BdG())
end
