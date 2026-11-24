"""
    EigenState

The basis state of an [`eigenbasis`](@ref) space: eigenstate number `index` of the space, in the sector `qn`.
"""
struct EigenState{Q} <: AbstractBasisState
    qn::Q
    index::Int
end
_eigen_qn(state::EigenState) = state.qn

# One atom whose basis states are the columns of `vectors`, given in the basis of `parent`. Operators act on it as on
# `parent`, so the atom takes the group of `parent`, which must be a single group: its basis states are superpositions
# of states of all its modes, and only operators on other groups commute with them without phase factors.
struct EigenbasisSpace{Q, P, M, E} <: AbstractAtomicHilbertSpace{EigenState{Q}}
    parent::P
    vectors::M
    energies::Vector{E}
    qns::Vector{Q}
end

"""
    eigenbasis(ham, H)

The space spanned by the eigenstates of the Hermitian `ham`, a symbolic operator or its matrix on `H`, found sector by
sector of `H` and ordered by increasing energy within each sector. It has the sectors of `H`, and `ham` must not
couple them. `representation(op, Heig)` is `S' * M * S` for the matrix `M` of `op` on `H` and the eigenvectors `S`,
and `change_basis(ρ, Heig => H)` is `S * ρ * S'`. The space works as a factor of `tensor_product` and
[`open_system`](@ref) together with spaces of other groups, but not with other modes of the group of `H`. See
[`energies`](@ref).

```julia
Heig = eigenbasis(ham, hilbert_space(f, 1:2, NumberConservation()))
Hlr, Hl, Hr, left, right = open_system(Heig)
L = representation(-1im * (left(ham) - right(ham)) + dissipator(f[1]), Hlr)
```
"""
function eigenbasis(ham, H::AbstractHilbertSpace)
    length(groups(H)) == 1 || throw(ArgumentError("The eigenbasis of a space with several groups is not supported, got the groups $(group_ids(H))."))
    M = ham isa AbstractMatrix ? ham : representation(ham, H)
    size(M) == (dim(H), dim(H)) || throw(DimensionMismatch("The matrix of `ham` must have the size $((dim(H), dim(H))), got $(size(M))."))
    qns = collect(quantumnumbers(H))
    blocks = [indices(qn, H) for qn in qns]
    _check_block_diagonal(M, blocks)
    decompositions = map(inds -> _hermitian_eigen(M[inds, inds]), blocks)
    rows = reduce(vcat, [repeat(inds, length(F.values)) for (inds, F) in zip(blocks, decompositions)])
    cols = reduce(vcat, [repeat(inds; inner = length(inds)) for inds in blocks])
    S = sparse(rows, cols, reduce(vcat, [vec(F.vectors) for F in decompositions]), dim(H), dim(H))
    Heig = EigenbasisSpace(H, S, reduce(vcat, [F.values for F in decompositions]), reduce(vcat, [fill(qn, length(inds)) for (qn, inds) in zip(qns, blocks)]))
    H isa SectorHilbertSpace || return Heig
    return sector_space(Heig, collect(basisstates(Heig)), _eigen_qn, SectorConstraint(_eigen_qn))
end
function _check_block_diagonal(M, blocks)
    block = zeros(Int, size(M, 1))
    foreach(((n, inds),) -> block[inds] .= n, enumerate(blocks))
    tol = _roundoff(M)
    for (i, j, v) in zip(findnz(sparse(M))...)
        block[i] == block[j] || abs(v) <= tol || throw(ArgumentError("The Hamiltonian couples the sectors of the space, so it has no eigenbasis sector by sector."))
    end
end
function _hermitian_eigen(A)
    A = Matrix(A)
    norm(A - A') <= _roundoff(A) || throw(ArgumentError("The Hamiltonian must be Hermitian."))
    return LinearAlgebra.eigen(Hermitian(A))
end
_roundoff(M) = isempty(M) ? 0.0 : 100 * eps(float(real(eltype(M)))) * size(M, 1) * maximum(abs, M)

"""
    energies(Heig)

The energies of the basis states of an [`eigenbasis`](@ref) space or of a sector of one.
"""
energies(H::EigenbasisSpace) = H.energies
energies(H::AbstractSubspace{<:EigenState, <:EigenbasisSpace}) = [parent(H).energies[state.index] for state in basisstates(H)]

const _EigenSpaces = Union{EigenbasisSpace, AbstractSubspace{<:EigenState, <:EigenbasisSpace}}
_eigen_root(H::EigenbasisSpace) = H
_eigen_root(H::AbstractSubspace) = parent(H)

Base.parent(H::EigenbasisSpace) = H.parent
dim(H::EigenbasisSpace) = length(H.energies)
basisstate(n::Integer, H::EigenbasisSpace) = EigenState(H.qns[n], Int(n))
basisstates(H::EigenbasisSpace) = BasisStates(H, Iterators.map(n -> basisstate(n, H), 1:dim(H)))
_state_index(state::EigenState, H::EigenbasisSpace) = checkbounds(Bool, H.qns, state.index) && isequal(H.qns[state.index], state.qn) ? state.index : 0
interpret_state(state::EigenState, ::EigenbasisSpace) = state
interpret_state(state, H::EigenbasisSpace) = throw(ArgumentError("`$state` is not a basis state of `$H`. Represent it on the parent space `H` and use `change_basis(v, H => Heig)`."))
has_arithmetic_state_index(::EigenbasisSpace) = true
atomic_id(H::EigenbasisSpace) = (:eigenbasis, atom_ids(H.parent)...)
group_id(H::EigenbasisSpace) = group_id(only(groups(H.parent)))
add_tag(H::EigenbasisSpace, tag) = EigenbasisSpace(add_tag(H.parent, tag), H.vectors, H.energies, H.qns)
# The transposed space has the complex conjugate basis, so that its matrices are the transposes.
TransposedSpace(H::EigenbasisSpace) = EigenbasisSpace(TransposedSpace(H.parent), conj(H.vectors), H.energies, H.qns)
Base.:(==)(H1::EigenbasisSpace, H2::EigenbasisSpace) = H1 === H2 || (H1.parent == H2.parent && H1.vectors == H2.vectors)
Base.hash(H::EigenbasisSpace, h::UInt) = hash((H.parent, H.vectors), h)
Base.show(io::IO, H::EigenbasisSpace) = print(io, "$(dim(H))-dimensional eigenbasis of ", H.parent)
_amplitude_eltype(H::EigenbasisSpace) = eltype(H.vectors)

# An eigenbasis is a group of its own. Joining it with other modes of its group would need the fermionic phase factors
# of each Fock state in an eigenstate, which a basis state of the atom does not have.
combine_into_group(::Group{Fermion}, atoms::AbstractVector{<:EigenbasisSpace}) = only(atoms)
combine_into_group(::Group{Majorana}, atoms::AbstractVector{<:EigenbasisSpace}) = only(atoms)
mode(H::EigenbasisSpace) = throw(ArgumentError("`$H` cannot be combined with other modes of its group. Build the eigenbasis on all of them, or put the other modes in another field."))

# The matrix of `op` in the eigenbasis `root`. Entries at the level of the roundoff are dropped, so that conserved
# quantities are diagonal.
function _eigen_matrix(op, root::EigenbasisSpace; projection = false)
    M = representation(op, root.parent; projection)
    return droptol!(root.vectors' * M * root.vectors, _roundoff(M))
end
_eigen_indices(H::EigenbasisSpace) = 1:dim(H)
_eigen_indices(H::AbstractSubspace) = [state.index for state in basisstates(H)]
function _eigen_matrix(op, Hout, Hin, repr; projection = false)
    A = _eigen_matrix(op, _eigen_root(Hin); projection)[:, _eigen_indices(Hin)]
    rows = _eigen_indices(Hout)
    projection || iszero(A[setdiff(axes(A, 1), rows), :]) || _skip_outside(0, false)
    return repr isa EagerDenseRepr ? Matrix(A[rows, :]) : A[rows, :]
end
const _EigenAxes = Axes{<:Tuple{_EigenSpaces, _EigenSpaces}}
for T in (NCMul, NCAdd)
    @eval _matrix_representation(op::$T, bases, space::Union{_EigenSpaces, _EigenAxes}, repr::Union{EagerSparseRepr, EagerDenseRepr}, chunking; projection = false, kwargs...) =
        _eigen_matrix(op, _output_space(space), _input_space(space), repr; projection)
end
_term_matrix_representation(op::NCMul, H::EigenbasisSpace, repr::Union{EagerSparseRepr, EagerDenseRepr}, chunking; projection = false, kwargs...) =
    _eigen_matrix(op, H, H, repr; projection)

# On a subspace of a product, an operator maps an eigenstate to a superposition, so the matrix of each eigenbasis group
# is applied column by column instead of state by state.
struct _EigenAction{M}
    matrix::M
end
precomputation_before_operator_application(op::NCMul, H::EigenbasisSpace) = _EigenAction(_eigen_matrix(op, H))
_eigen_outputs(state, root, action) = ((basisstate(rowvals(action.matrix)[p], root), nonzeros(action.matrix)[p]) for p in nzrange(action.matrix, state.index))
_apply_local_operators(op, ::EigenState, ::EigenbasisSpace, precomp) = throw(ArgumentError("An operator maps an eigenstate to a superposition, so only eager matrix representations are supported on an eigenbasis."))

# A product over groups, some of which are eigenbases, gives the product of the outputs of each group.
function chunked_operator_indices_and_amplitudes_generic!(accumulator, op::ProductOperator, space::AbstractHilbertSpace, precomps::Tuple, input_inds, col_offset; projection)
    any(p -> p isa _EigenAction, precomps) || return invoke(chunked_operator_indices_and_amplitudes_generic!, Tuple{Any, Any, AbstractHilbertSpace, Any, Any, Any}, accumulator, op, space, precomps, input_inds, col_offset; projection)
    return _push_outputs!(state -> _product_outputs(op, state, precomps), accumulator, space, input_inds, col_offset, projection)
end
function operator_indices_and_amplitudes_generic!(accumulator, op::ProductOperator, space, precomps::Tuple; projection)
    any(p -> p isa _EigenAction, precomps) || return invoke(operator_indices_and_amplitudes_generic!, Tuple{Any, Any, Any, Any}, accumulator, op, space, precomps; projection)
    return chunked_operator_indices_and_amplitudes_generic!(accumulator, op, space, precomps, 1:dim(_input_space(space)), 0; projection)
end
function _product_outputs(op, state::ProductState, precomps)
    outcomes = [(state.states, true)]
    for (part, group, precomp, n) in zip(op.ops, op.spaces, precomps, op.inds)
        outcomes = [(Base.setindex(states, s, n), amp * a) for (states, amp) in outcomes for (s, a) in _group_outputs(part, states[n], group, precomp)]
    end
    return ((typeof(state)(states), amp) for (states, amp) in outcomes)
end
_group_outputs(part, state, group, action::_EigenAction) = _eigen_outputs(state, group, action)
_group_outputs(part, state, group, precomp) = (_apply_local_operators(part, state, group, precomp),)

function _push_outputs!(outputs, accumulator, space, input_inds, col_offset, projection)
    Hin, Hout = _input_space(space), _output_space(space)
    for inind in input_inds
        state = basisstate(inind, Hin)
        for (newstate, amp) in outputs(state)
            iszero(amp) && continue
            outind = state_index(newstate, Hout)
            _skip_outside(outind, projection, newstate, state, Hout) || push_inds_amps!(accumulator, outind, inind - col_offset, amp)
        end
    end
    return accumulator
end

# Operator constraints: a conserved quantity is diagonal in the eigenbasis only as a sum, so its values are those of
# the sum.
_acts_in(x, H::_EigenSpaces) = symbolic_group(x) == group_id(H)
function _diagonal_values(weighted_terms, H::_EigenSpaces)
    foreach(((w, t),) -> _check_exact(w, t), weighted_terms)
    op = sum(((w, t),) -> w * NCMul(1, t.factors), weighted_terms)
    M = representation(op, H, Matrix)
    isdiag(M) || throw(ArgumentError("A constraint operator must be diagonal in the eigenbasis, but `$op` is not."))
    values = map(v -> _exact_value(v, op), diag(M))
    all(isreal, values) || throw(ArgumentError("Constraint operators with complex values are not supported, but `$op` has complex values. $_ZN_HINT"))
    return Rational{Int}.(real.(values))
end

# Moving between an eigenbasis and the space it was built on
complementary_subsystem(H::AbstractHilbertSpace, Hsub::_EigenSpaces) =
    atomic_id(_eigen_root(Hsub)) in atom_ids(H) ? invoke(complementary_subsystem, Tuple{AbstractHilbertSpace, Any}, H, Hsub) : nothing
function basis_transformation(Hfrom::_EigenSpaces, Hto::AbstractHilbertSpace)
    root = _eigen_root(Hfrom)
    U, dropped = basis_transformation(root.parent, Hto)
    S = root.vectors[:, _eigen_indices(Hfrom)]
    return U * S, findall(j -> !iszero(S[dropped, j]), axes(S, 2))
end
basis_transformation(Hfrom::AbstractHilbertSpace, Hto::_EigenSpaces) = (sparse(first(basis_transformation(Hto, Hfrom))'), Int[])
function basis_transformation(Hfrom::_EigenSpaces, Hto::_EigenSpaces)
    _eigen_root(Hfrom) == _eigen_root(Hto) && return invoke(basis_transformation, Tuple{Any, Any}, Hfrom, Hto)
    U, dropped = basis_transformation(Hfrom, _eigen_root(Hto).parent)
    return first(basis_transformation(_eigen_root(Hto).parent, Hto)) * U, dropped
end
# The eigenbasis of a symmetric sector is not supported, so these only resolve ambiguities.
basis_transformation(Hfrom::_EigenSpaces, Hto::_OrbitSpace) = throw(ArgumentError("Cannot change basis between an eigenbasis and a symmetric sector."))
basis_transformation(Hfrom::_OrbitSpace, Hto::_EigenSpaces) = throw(ArgumentError("Cannot change basis between an eigenbasis and a symmetric sector."))
_check_projection(Hfrom::AbstractHilbertSpace, Hto::_EigenSpaces) = Hfrom isa _EigenSpaces || dim(Hfrom) == dim(Hto) ||
    throw(ArgumentError("A state of `$Hfrom` is in general not in `$Hto`. Pass `projection = true` to project it."))

@testitem "Eigenbasis" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: eigenbasis, energies, change_basis, Axes
    @fermions f
    @bosons b
    ham = sum(0.3k * f[k]'f[k] for k in 1:3) + sum(0.5 * f[k]'f[k + 1] + hc for k in 1:2) + 0.7 * f[1]'f[1] * f[2]'f[2]
    H = hilbert_space(f, 1:3, NumberConservation())
    Heig = eigenbasis(ham, H)
    M = representation(ham, H)
    Fs = [eigen(Hermitian(Matrix(M[indices(n, H), indices(n, H)]))) for n in quantumnumbers(H)]
    S = cat([F.vectors for F in Fs]...; dims = (1, 2))
    @test quantumnumbers(Heig) == quantumnumbers(H)
    @test energies(Heig) ≈ reduce(vcat, [F.values for F in Fs])
    for op in (ham, f[1], f[1]' * f[2]' * f[3] + 0.2im * f[2]'f[1])
        @test representation(op, Heig) ≈ S' * representation(op, H) * S
    end
    @test representation(f[1], Axes(sector(1, Heig), sector(2, Heig))) ≈ (S' * representation(f[1], H) * S)[indices(1, Heig), indices(2, Heig)]
    ψ = randn(ComplexF64, dim(H))
    @test change_basis(ψ, Heig => H) ≈ S * ψ

    # A factor of a tensor product with other groups
    H0 = hilbert_space(f, 1:2)
    ham0 = 0.3 * f[1]'f[1] + (0.5im * f[1]'f[2] + 0.4 * f[1]'f[2]' + hc)
    S0 = eigen(Hermitian(Matrix(representation(ham0, H0)))).vectors
    Hb = hilbert_space(b[1], 3)
    op = f[1]' * b[1] + hc + b[1]'b[1] * f[2]'f[2]
    U = kron(I(3), S0)
    @test representation(op, tensor_product(eigenbasis(ham0, H0), Hb)) ≈ U' * representation(op, tensor_product(H0, Hb)) * U

    @test_throws ArgumentError eigenbasis(ham0, H)
    @test_throws ArgumentError tensor_product(Heig, hilbert_space(f, 4:4))
    @test_throws ArgumentError representation(f[1], Heig, :lazy) * ψ
end

@testitem "Lindbladian in an eigenbasis" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: eigenbasis, change_basis, open_system, dissipator
    @fermions f
    ham = sum(0.3k * f[k]'f[k] for k in 1:3) + sum(0.5 * f[k]'f[k + 1] + hc for k in 1:2) + 0.7 * f[1]'f[1] * f[2]'f[2]
    Ntot = sum(f[k]'f[k] for k in 1:3)
    H = hilbert_space(f, 1:3, NumberConservation())
    Heig = eigenbasis(ham, H)
    jumps = [0.4 * f[1], 0.3 * f[3]']
    lindbladian(left, right) = -1im * (left(ham) - right(ham)) + sum(dissipator, jumps)
    spectrum(L) = sort(eigvals(Matrix(L)); by = x -> round.(reim(x); digits = 8))
    function steady_state(L, Hlr, Hl, Hr)
        ρ = reshape(nullspace(Matrix(L))[:, 1], Hlr => (Hl, Hr))
        return ρ / tr(ρ)
    end
    Hlr, Hl, Hr, left, right = open_system(H)
    Elr, El, Er = open_system(Heig)
    for (Hd, Ed) in ((Hlr, Elr), (tensor_product((Hl, Hr); constraint = left(Ntot) - right(Ntot) => 0), tensor_product((El, Er); constraint = left(Ntot) - right(Ntot) => 0)))
        L, Le = representation(lindbladian(left, right), Hd), representation(lindbladian(left, right), Ed)
        @test spectrum(Le) ≈ spectrum(L)
        @test change_basis(steady_state(Le, Ed, El, Er), Heig => H) ≈ steady_state(L, Hd, Hl, Hr)
    end

    # From matrices on the eigenbasis
    h, c = representation(ham, Heig), representation(jumps[1], Heig)
    sandwich(A, B) = tensor_product((A, transpose(B)), (El, Er) => Elr)
    L = -1im * (embed(h, El => Elr) - embed(transpose(h), Er => Elr)) + sandwich(c, c') - (embed(c'c, El => Elr) + embed(transpose(c'c), Er => Elr)) / 2
    @test L ≈ representation(-1im * (left(ham) - right(ham)) + dissipator(jumps[1]), Elr)

    # The populations: the Pauli rate matrix with rates |⟨a|J|b⟩|² from b to a
    populations = tensor_product((El, Er); constraint = s -> s.states[1].index == s.states[2].index)
    R = sum(J -> abs2.(representation(J, Heig, Matrix)), jumps)
    @test representation(lindbladian(left, right), populations; projection = true) ≈ R - Diagonal(vec(sum(R; dims = 1)))
end
