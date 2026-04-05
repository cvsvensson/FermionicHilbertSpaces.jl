

"""
    tensor_product(spaces; constraint=NoSymmetry())

Construct the composite Hilbert space from the spaces in `spaces`, with optional keyword argument `constraint`.
"""
function tensor_product_no_constraints(spaces)
    atoms = Iterators.flatten(Iterators.map(atomic_factors, spaces))
    groups = groupby(cluster_id, atoms; sort=false)

    clusters = map((id, atoms) -> combine_into_cluster(id, atoms), keys(groups), values(groups))
    space = if length(clusters) == 1
        only(clusters)
    else
        ProductSpace(Tuple(clusters), collect(atoms))
    end
    if any(isconstrained, spaces)
        mapper = state_mapper(space, spaces)
        states = _find_combined_states(space, spaces, mapper)
        return ConstrainedSpace(space, states)
    end
    return space
end
tensor_product(spaces...; kwargs...) = tensor_product(spaces; kwargs...)
function tensor_product(spaces; constraint=NoSymmetry())
    if length(spaces) == 1
        return constrain_space(only(spaces), constraint)
    end
    full_space = tensor_product_no_constraints(spaces)
    constraint == NoSymmetry() && return full_space

    states = supports_branch_pruning(constraint) ? generate_states(spaces, constraint, full_space) : basisstates(full_space)
    if supports_sector_grouping(constraint) && supports_filtering(constraint)
        return block_space(full_space, states, sector_function(constraint, full_space))
    elseif supports_filtering(constraint)
        filtered_states = collect(Iterators.filter(filter_function(constraint, full_space), states))
        return ConstrainedSpace(full_space, filtered_states)
    else
        return ConstrainedSpace(full_space, states)
    end
end

function check_tensor_product_basis_compatibility(b1::AbstractHilbertSpace, b2::AbstractHilbertSpace, b3::AbstractHilbertSpace)
    if vcat(collect(keys(b1)), collect(keys(b2))) != collect(keys(b3))
        throw(ArgumentError("The labels of the output basis are not the same (or ordered the same) as the labels of the input bases. $(keys(b1)) * $(keys(b2)) != $(keys(b3))"))
    end
end

size_compatible(m, H) = size(m) == size(m) == ntuple(_ -> dim(H), ndims(m))
size_compatible(m::UniformScaling, H) = true
kron_sizes_compatible(ms, Hs) = all(size_compatible(m, H) for (m, H) in zip(ms, Hs))

@testitem "Kron size compatibility and error handling" begin
    using LinearAlgebra, Random
    import FermionicHilbertSpaces: kron_sizes_compatible
    Random.seed!(1234)
    @fermions a
    H1 = hilbert_space(a, 1:2)
    H2 = hilbert_space(a, 3:3)
    H3 = hilbert_space(a, 4:6)
    Hs = [H1, H2, H3]
    ms = m1, m2, m3 = [rand(dim(H), dim(H)) for H in Hs]
    vs = v1, v2, v3 = [rand(dim(H)) for H in Hs]
    @test kron_sizes_compatible(ms, Hs)
    @test kron_sizes_compatible([I, I, m3], Hs)
    @test !kron_sizes_compatible(ms, reverse(Hs))
    @test !kron_sizes_compatible([I, I, m2], Hs)
    @test kron_sizes_compatible(vs, Hs)
    @test !kron_sizes_compatible(vs, reverse(Hs))
    @test_throws ArgumentError generalized_kron(ms, reverse(Hs))
    H12 = tensor_product(H1, H2)
    m_too_large = rand(dim(H12) + 1, dim(H12) + 1)
    @test_throws ArgumentError partial_trace(m_too_large, H12 => H1)
end

"""
    generalized_kron(ms, Hs, H::AbstractHilbertSpace=tensor_product(Hs))

Compute the tensor product of matrices or vectors in `ms` with respect to the spaces `Hs`, respectively. Return a matrix in the space `H`, which defaults to the tensor_product product of `Hs`.
"""
function generalized_kron(ms, Hs, H::AbstractHilbertSpace=tensor_product(Hs); kwargs...)
    kron_sizes_compatible(ms, Hs) || throw(ArgumentError("The sizes of `ms` must match the sizes of `Hs`"))
    N = ndims(first(ms))
    mout = allocate_tensor_product_result(ms, Hs, H)
    extend_state = state_mapper(H, Hs)
    if N == 1
        return generalized_kron_vec!(mout, Tuple(ms), Tuple(Hs), H, extend_state; kwargs...)
    elseif N == 2
        return generalized_kron_mat!(mout, Tuple(ms), Tuple(Hs), H, extend_state; kwargs...)
    end
    throw(ArgumentError("Only 1D or 2D arrays are supported"))
end

generalized_kron(Hs::Pair; kwargs...) = (ms...) -> generalized_kron(ms, Hs; kwargs...)
generalized_kron(ms, Hs::Pair; kwargs...) = generalized_kron(ms, first(Hs), last(Hs); kwargs...)

uniform_to_sparse_type(::Type{UniformScaling{T}}) where {T} = SparseMatrixCSC{T,Int}
uniform_to_sparse_type(::Type{T}) where {T} = T
function allocate_tensor_product_result(ms, Hs, H)
    T = Base.promote_eltype(ms...)
    N = ndims(first(ms))
    types = map(uniform_to_sparse_type ∘ typeof, ms)
    MT = Base.promote_op(kron, types...)
    Nout = dim(H)
    _mout = Zeros(T, ntuple(j -> Nout, N))
    try
        convert(MT, _mout)
    catch
        Array(_mout)
    end
end

tensor_product_iterator(m::SparseArrays.AbstractSparseVecOrMat, ::AbstractHilbertSpace) = zip(findnz(m)[1:2]...)
tensor_product_iterator(m::AbstractArray, ::AbstractHilbertSpace) = CartesianIndices(m)
tensor_product_iterator(::UniformScaling, H::AbstractHilbertSpace) = diagind(I(length(basisstates(H))), IndexCartesian())

function generalized_kron_mat!(mout::AbstractMatrix{T}, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state; phase_factors::Bool=true, skipmissing=false) where T
    fill!(mout, zero(T))
    inds = Base.product(map(tensor_product_iterator, ms, Hs)...)
    pfh = phase_factors ? kron_phase_factor(extend_state) : (f1, f2) -> 1
    for I in inds
        I1 = map(i -> i[1], I)
        I2 = map(i -> i[2], I)
        state1 = map(basisstate, I1, Hs)
        state2 = map(basisstate, I2, Hs)
        fullstates1, amps1 = combine_states(state1, extend_state)
        fullstates2, amps2 = combine_states(state2, extend_state)
        for (fullstate1, w1) in zip(fullstates1, amps1)
            outind1 = state_index(fullstate1, H)
            if ismissing(outind1)
                skipmissing && continue
                throw(ArgumentError("The state $fullstate1 does not exist in the full Hilbert space"))
            end
            for (fullstate2, w2) in zip(fullstates2, amps2)
                outind2 = state_index(fullstate2, H)
                if ismissing(outind2)
                    skipmissing && continue
                    throw(ArgumentError("The state $fullstate2 does not exist in the full Hilbert space"))
                end
                s = pfh(fullstate1, fullstate2)
                v = prod(ntuple(i -> ms[i][I1[i], I2[i]], length(ms)))
                mout[outind1, outind2] += w1 * w2 * v * s
            end
        end
    end
    return mout
end

function generalized_kron_mat!(mout::SparseMatrixCSC{T}, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state; phase_factors::Bool=true, skipmissing=false) where T
    # phase_factors && (isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be consistent with the jordan-wigner ordering of the full system")))
    # !phase_factors && (ispartition(Hs, H) || throw(ArgumentError("The subsystems must be a partition of the full system")))
    inds = Base.product(map(tensor_product_iterator, ms, Hs)...)

    pfh = phase_factors ? kron_phase_factor(extend_state) : (f1, f2) -> 1
    Is, Js, Vs = Int[], Int[], T[]
    sizehint!(Is, length(inds))
    sizehint!(Js, length(inds))
    sizehint!(Vs, length(inds))
    for I in inds
        I1 = map(i -> i[1], I)
        I2 = map(i -> i[2], I)
        state1 = map(basisstate, I1, Hs)
        state2 = map(basisstate, I2, Hs)
        fullstates1, amps1 = combine_states(state1, extend_state)
        fullstates2, amps2 = combine_states(state2, extend_state)
        for (fullstate1, w1) in zip(fullstates1, amps1)
            outind1 = state_index(fullstate1, H)

            if ismissing(outind1)
                skipmissing && continue
                throw(ArgumentError("The state $fullstate1 does not exist in the full Hilbert space"))
            end
            for (fullstate2, w2) in zip(fullstates2, amps2)
                outind2 = state_index(fullstate2, H)
                if ismissing(outind2)
                    skipmissing && continue
                    throw(ArgumentError("The state $fullstate2 does not exist in the full Hilbert space"))
                end
                s = pfh(fullstate1, fullstate2)

                v = prod(ntuple(i -> ms[i][I1[i], I2[i]], length(ms)))
                push!(Is, outind1)
                push!(Js, outind2)
                push!(Vs, w1 * w2 * v * s)
            end
        end
    end
    return mout .= sparse(Is, Js, Vs, size(mout, 1), size(mout, 2))
end

function generalized_kron_vec!(mout, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state; phase_factors=true)
    fill!(mout, zero(eltype(mout)))
    U = embedding_unitary(Hs, H)
    dimlengths = map(length ∘ basisstates, Hs)
    inds = CartesianIndices(Tuple(dimlengths))
    for I in inds
        TI = Tuple(I)
        fock = map(basisstate, TI, Hs)
        states, amps = combine_states(fock, extend_state)
        for (fullfock, w) in zip(states, amps)
            outind = state_index(fullfock, H)
            mout[outind] += w * mapreduce((i1, m) -> m[i1], *, TI, ms)
        end
    end
    return U * mout
end
embedding_unitary(Hs, H::ProductSpace{Nothing}) = I

"""
    tensor_product(ms, Hs, H::AbstractHilbertSpace; kwargs...)

Compute the ordered product of the fermionic embeddings of the matrices `ms` in the spaces `Hs` into the space `H`.
`kwargs` can be passed a bool `phase_factors`.
"""
function tensor_product(ms::Union{<:AbstractVector,<:Tuple}, Hs, H::AbstractHilbertSpace; kwargs...)
    # See eq. 26 in J. Phys. A: Math. Theor. 54 (2021) 393001
    # isorderedpartition(Hs, H) || throw(ArgumentError("The subsystems must be a partition consistent with the jordan-wigner ordering of the full system"))
    return mapreduce(((m, fine_basis),) -> embed(m, fine_basis, H, kwargs...), *, zip(ms, Hs))
end
tensor_product(ms::Union{<:AbstractVector,<:Tuple}, HsH::Pair{<:Any,<:AbstractHilbertSpace}; kwargs...) = tensor_product(ms, first(HsH), last(HsH); kwargs...)
tensor_product(HsH::Pair{<:Any,<:AbstractHilbertSpace}; kwargs...) = (ms...) -> tensor_product(ms, first(HsH), last(HsH); kwargs...)

struct PhaseMap{F}
    phases::Matrix{Int}
    fockstates::Vector{F}
end
struct LazyPhaseMap{M,F} <: AbstractMatrix{Int}
    fockstates::Vector{F}
end
Base.length(p::LazyPhaseMap) = length(p.fockstates)
Base.ndims(::LazyPhaseMap) = 2
function Base.size(p::LazyPhaseMap, d::Int)
    d < 1 && error("arraysize: dimension out of range")
    d in (1, 2) ? length(p.fockstates) : 1
end
Base.size(p::LazyPhaseMap) = (length(p.fockstates), length(p.fockstates))
function Base.show(io::IO, p::LazyPhaseMap{M,F}) where {M,F}
    print(io, "LazyPhaseMap{$M,$F}(")
    show(io, p.fockstates)
    print(")")
end
Base.show(io::IO, ::MIME"text/plain", p::LazyPhaseMap) = show(io, p)
Base.getindex(p::LazyPhaseMap{M}, n1::Int, n2::Int) where {M} = phase_factor_f(p.fockstates[n1], p.fockstates[n2], M)
function phase_map(fockstates, M::Int)
    phases = zeros(Int, length(fockstates), length(fockstates))
    for (n1, f1) in enumerate(fockstates)
        for (n2, f2) in enumerate(fockstates)
            phases[n1, n2] = phase_factor_f(f1, f2, M)
        end
    end
    PhaseMap(phases, fockstates)
end
phase_map(N::Int) = phase_map(map(FockNumber, UnitRange{UInt64}(0, 2^N - 1)), N)
phase_map(H::AbstractHilbertSpace) = phase_map(collect(basisstates(H)), nbr_of_modes(H))
function LazyPhaseMap(N::Int)
    states = map(FockNumber, UnitRange{UInt64}(0, 2^N - 1))
    LazyPhaseMap{N,eltype(states)}(states)
end
SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(::LazyPhaseMap, rest...) = SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(rest...)
(p::PhaseMap)(op::AbstractMatrix) = p.phases .* op
(p::LazyPhaseMap)(op::AbstractMatrix) = p .* op
@testitem "Phase map: sign pattern structure" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: fermions
    # see App 2 in https://arxiv.org/pdf/2006.03087
    ns = 1:4
    phis = Dict(zip(ns, FermionicHilbertSpaces.phase_map.(ns)))
    lazyphis = Dict(zip(ns, FermionicHilbertSpaces.LazyPhaseMap.(ns)))
    @test all(sum(phis[n].phases .== -1) == (2^n - 2) * 2^n / 2 for n in ns)
    @test all(sum(phis[n].phases .== -1) == (2^n - 2) * 2^n / 2 for n in ns)
    @fermions f
    for N in ns
        H = hilbert_space(f, 1:N)
        c = fermions(H)
        # Now let's make commuting fermions (hardcore bosons)
        q = Dict(k => embed(only(fermions(hilbert_space(f[k])))[2], hilbert_space(f[k]) => H; phase_factors=false) for k in 1:N)
        @test all(map(n -> q[n] == phis[N](c[n]), 1:N))
        c2 = map(n -> phis[N](c[n]), 1:N)
        @test phis[N](phis[N](c[1])) == c[1]
        # c is fermionic
        @test all([c[n] * c[n2] == -c[n2] * c[n] for n in 1:N, n2 in 1:N])
        @test all([c[n]' * c[n2] == -c[n2] * c[n]' + I * (n == n2) for n in 1:N, n2 in 1:N])
        # c2 is hardcore bosons
        @test all([c2[n] * c2[n2] == c2[n2] * c2[n] for n in 1:N, n2 in 1:N])
        @test all([c2[n]' * c2[n2] == (-c2[n2] * c2[n]' + I) * (n == n2) + (n !== n2) * (c2[n2] * c2[n]') for n in 1:N, n2 in 1:N])
    end

    H1 = hilbert_space(f, 1:1)
    c1 = fermions(H1)
    H2 = hilbert_space(f, 2:2)
    c2 = fermions(H2)
    H12 = hilbert_space(f, 1:2)
    c12 = fermions(H12)
    p1 = FermionicHilbertSpaces.LazyPhaseMap(1)
    p2 = FermionicHilbertSpaces.phase_map(2)
    @test FermionicHilbertSpaces.fermionic_tensor_product_with_kron_and_maps((c1[1], I(2)), (p1, p1), p2) == c12[1]
    @test FermionicHilbertSpaces.fermionic_tensor_product_with_kron_and_maps((I(2), c2[2]), (p1, p1), p2) == c12[2]

    ms = (rand(2, 2), rand(2, 2))
    @test FermionicHilbertSpaces.fermionic_tensor_product_with_kron_and_maps(ms, (p1, p1), p2) == generalized_kron(ms, (H1, H2), H12)
end

function fermionic_tensor_product_with_kron_and_maps(ops, phis, phi)
    phi(kron(reverse(map((phi, op) -> phi(op), phis, ops))...))
end

function partial_trace(m, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), alg=default_partial_trace_alg(Hsub, H, complement), kwargs...)
    size_compatible(m, H) || throw(ArgumentError("The size of `m` must match the size of `H`"))
    if isnothing(complement)
        U = basis_transformation(Hsub, H)
        return U * m * U'
    end
    mout = zeros(eltype(m), dim(Hsub), dim(Hsub))
    partial_trace!(mout, m, H, Hsub, complement, alg; kwargs...)
end
function basis_transformation(H1, H2)
    #transforms from the basis of H1 to the basis of H2. Assumes they are the same basis states, just ordered differently. If they are not the same, this will throw an error.
    dim(H1) == dim(H2) || throw(ArgumentError("Hilbert spaces must have the same dimension for a basis transformation"))
    I = Vector{Int}(undef, dim(H1))
    J = Vector{Int}(undef, dim(H2))
    V = ones(Int, dim(H1))
    for (i, f1) in enumerate(basisstates(H1))
        j = state_index(f1, H2)
        if ismissing(j)
            throw(ArgumentError("The state $f1 in the first Hilbert space does not exist in the second Hilbert space"))
        end
        J[i] = j
        I[i] = i
    end
    return SparseArrays.sparse!(I, J, V)
end

"""
    partial_trace(m, H => Hsub; complement=complementary_subsystem(H, Hsub))

Compute the partial trace of `m` from `H` to `Hsub`. Fermionic phase factors are included if both `H` and `Hsub` are Fermionic, unless specified otherwise in `kwargs`.
"""
partial_trace(m, Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}; kwargs...) = partial_trace(m, Hs...; kwargs...)

abstract type AbstractPartialTraceAlg end
struct SubsystemPartialTraceAlg <: AbstractPartialTraceAlg end
struct FullPartialTraceAlg <: AbstractPartialTraceAlg end
default_partial_trace_alg(Hsub, H, Hcomp) = dim(Hsub)^2 * dim(Hcomp) < dim(H)^2 ? SubsystemPartialTraceAlg() : FullPartialTraceAlg()
default_partial_trace_alg(Hsub, H, ::Nothing) = dim(Hsub)^2 < dim(H)^2 ? SubsystemPartialTraceAlg() : FullPartialTraceAlg()
#TODO: FullPartialTraceAlg exploits sparsity of the matrix which should be taken into account.

"""
    partial_trace!(mout, m, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement, extend_state=StateExtender((Hsub, complement), H); skipmissing=true, phase_factors=true)

Compute the partial trace of `m` from `H` to `Hsub`. 
"""
function partial_trace!(mout, m, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement, ::SubsystemPartialTraceAlg, mapper=state_mapper(H, (Hsub, complement)); skipmissing=true, phase_factors=true)
    fill!(mout, zero(eltype(mout)))

    substates = basisstates(Hsub)
    barstates = basisstates(complement)
    for f1 in substates
        I1 = state_index(f1, Hsub)
        for f2 in substates
            s2 = phase_factors ? partial_trace_phase_factor(f1, f2, Hsub) : 1
            I2 = state_index(f2, Hsub)
            for fbar in barstates
                fullstates1, amps1 = combine_states((f1, fbar), mapper)
                fullstates2, amps2 = combine_states((f2, fbar), mapper)
                for (fullf1, w1) in zip(fullstates1, amps1)
                    J1 = state_index(fullf1, H)
                    if ismissing(J1)
                        skipmissing && continue
                        throw(ArgumentError("The state $fullf1 is not in the full Hilbert space"))
                    end
                    for (fullf2, w2) in zip(fullstates2, amps2)
                        J2 = state_index(fullf2, H)
                        if ismissing(J2)
                            skipmissing && continue
                            throw(ArgumentError("The state $fullf2 is not in the full Hilbert space"))
                        end
                        s1 = phase_factors ? partial_trace_phase_factor(fullf1, fullf2, H) : 1
                        s = s2 * s1
                        mout[I1, I2] += w1 * w2 * s * m[J1, J2]
                    end
                end
            end
        end
    end
    return mout
end

function partial_trace!(mout, m::AbstractMatrix, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement, ::FullPartialTraceAlg, mapper=state_mapper(H, (Hsub, complement)); phase_factors=true, skipmissing=false)
    fill!(mout, zero(eltype(mout)))
    inds = tensor_product_iterator(m, H)
    splitsamps = map(Base.Fix2(split_state, mapper), basisstates(H))
    for I in inds
        f1 = basisstate(I[1], H)
        f2 = basisstate(I[2], H)
        splits1, amps1 = splitsamps[I[1]]
        splits2, amps2 = splitsamps[I[2]]
        for ((f1sub, f1bar), w1) in zip(splits1, amps1)
            J1 = state_index(f1sub, Hsub)
            if ismissing(J1)
                skipmissing && continue
                throw(ArgumentError("The state $f1sub is not in the subsystem Hilbert space"))
            end
            for ((f2sub, f2bar), w2) in zip(splits2, amps2)
                f1bar != f2bar && continue
                J2 = state_index(f2sub, Hsub)
                if ismissing(J2)
                    skipmissing && continue
                    throw(ArgumentError("The state $f2sub is not in the subsystem Hilbert space"))
                end
                s1 = phase_factors ? partial_trace_phase_factor(f1, f2, H) : 1
                s2 = phase_factors ? partial_trace_phase_factor(f1sub, f2sub, Hsub) : 1
                s = s2 * s1
                mout[J1, J2] += w1 * w2 * s * m[I[1], I[2]]
            end
        end
    end
    return mout
end

"""
    partial_trace(H => Hsub; kwargs...)

Compute the partial trace map from `H` to `Hsub`, represented by a sparse matrix of dimension `dim(Hsub)^2 x dim(H)^2` that can be multiplied with a vectorized density matrix. 
"""
partial_trace(Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}; kwargs...) = partial_trace_map(Hs...; kwargs...)
function partial_trace_map(H, Hsub; complement=complementary_subsystem(H, Hsub), alg=default_partial_trace_alg(Hsub, H, complement), kwargs...)
    partial_trace_map(H, Hsub, complement, alg; kwargs...)
end
function partial_trace_map(H, Hsub, complement, ::SubsystemPartialTraceAlg, mapper=state_mapper(H, (Hsub, complement)); skipmissing=true, phase_factors=true)
    # we use skipmissing = true here as the default, since there is if H has constraints, there may be combinations of states from Hsub and the complement that do not exist in H, even if every split state from H exists in Hsub and the complement.
    substates = basisstates(Hsub)
    barstates = basisstates(complement)
    indI = LinearIndices((1:dim(Hsub), 1:dim(Hsub)))
    indJ = LinearIndices((1:dim(H), 1:dim(H)))
    Is = Int[]
    Js = Int[]
    Vs = Int[]
    for f1 in substates, f2 in substates
        s2 = phase_factors ? partial_trace_phase_factor(f1, f2, Hsub) : 1
        I1 = state_index(f1, Hsub)
        I2 = state_index(f2, Hsub)
        for fbar in barstates
            fullstates1, amps1 = combine_states((f1, fbar), mapper)
            fullstates2, amps2 = combine_states((f2, fbar), mapper)
            for (fullf1, w1) in zip(fullstates1, amps1)
                J1 = state_index(fullf1, H)
                if ismissing(J1)
                    skipmissing && continue
                    throw(ArgumentError("The state $fullf1 is not in the full Hilbert space."))
                end
                for (fullf2, w2) in zip(fullstates2, amps2)
                    J2 = state_index(fullf2, H)
                    if ismissing(J2)
                        skipmissing && continue
                        throw(ArgumentError("The state $fullf2 is not in the full Hilbert space."))
                    end
                    s1 = phase_factors ? partial_trace_phase_factor(fullf1, fullf2, H) : 1
                    s = s2 * s1
                    push!(Is, indI[I1, I2])
                    push!(Js, indJ[J1, J2])
                    push!(Vs, w1 * w2 * s)
                end
            end
        end
    end
    return sparse(Is, Js, Vs, dim(Hsub)^2, dim(H)^2)
end

function partial_trace_map(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement, ::FullPartialTraceAlg, mapper=state_mapper(H, (Hsub, complement)); skipmissing=false, phase_factors=true)
    states = basisstates(H)
    indI = LinearIndices((1:dim(Hsub), 1:dim(Hsub)))
    indJ = LinearIndices((1:dim(H), 1:dim(H)))
    Is = Int[]
    Js = Int[]
    Vs = Int[]
    substates2 = map(Base.Fix2(split_state, mapper), states)
    for f1 in states
        J1 = state_index(f1, H)
        splits1, amps1 = split_state(f1, mapper)
        for ((f1sub, f1bar), w1) in zip(splits1, amps1)
            I1 = state_index(f1sub, Hsub)
            if ismissing(I1)
                skipmissing && continue
                throw(ArgumentError("The state $f1sub is not in the subsystem Hilbert space"))
            end
            for (f2, (splits2, amps2)) in zip(states, substates2)
                for ((f2sub, f2bar), w2) in zip(splits2, amps2)
                    if f1bar != f2bar
                        continue
                    end
                    s1 = phase_factors ? partial_trace_phase_factor(f1, f2, H) : 1
                    s2 = phase_factors ? partial_trace_phase_factor(f1sub, f2sub, Hsub) : 1
                    s = s2 * s1
                    I2 = state_index(f2sub, Hsub)
                    if ismissing(I2)
                        skipmissing && continue
                        throw(ArgumentError("The state $f2sub is not in the subsystem Hilbert space"))
                    end
                    J2 = state_index(f2, H)
                    push!(Is, indI[I1, I2])
                    push!(Js, indJ[J1, J2])
                    push!(Vs, s * w1 * w2)
                end
            end
        end
    end
    return sparse(Is, Js, Vs, dim(Hsub)^2, dim(H)^2)
end

function project_on_parities(op::AbstractMatrix, H, Hs, parities)
    length(Hs) == length(parities) || throw(ArgumentError("The number of parities must match the number of subsystems"))
    for (bsub, parity) in zip(Hs, parities)
        op = project_on_subparity(op, H, bsub, parity)
    end
    return op
end

function project_on_subparity(op::AbstractMatrix, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, parity)
    P = embed(parityoperator(Hsub), Hsub, H)
    return project_on_parity(op, P, parity)
end

project_on_parity(op::AbstractMatrix, H::AbstractHilbertSpace, parity) = project_on_parity(op, parityoperator(H), parity)

function project_on_parity(op::AbstractMatrix, P::AbstractMatrix, parity)
    Peven = (I + P) / 2
    Podd = (I - P) / 2
    if parity == 1
        return Peven * op * Peven + Podd * op * Podd
    elseif parity == -1
        return Podd * op * Peven + Peven * op * Podd
    else
        throw(ArgumentError("Parity must be either 1 or -1"))
    end
end

@testitem "Parity projection" begin
    import FermionicHilbertSpaces: project_on_parity, project_on_parities
    @fermions f
    Hs = [hilbert_space(f, 2k-1:2k) for k in 1:3]
    H = tensor_product(Hs)
    op = rand(ComplexF64, dim(H), dim(H))
    local_parity_iter = (1, -1)
    all_parities = Base.product([local_parity_iter for _ in 1:length(Hs)]...)
    @test sum(project_on_parities(op, H, Hs, parities) for parities in all_parities) ≈ op

    ops = [rand(ComplexF64, dim(H), dim(H)) for H in Hs]
    for parities in all_parities
        projected_ops = [project_on_parity(op, Hsub, parity) for (op, Hsub, parity) in zip(ops, Hs, parities)]
        local op = tensor_product(projected_ops, Hs, H)
        @test op ≈ project_on_parities(op, H, Hs, parities)
    end
end

@testitem "Partial trace map" begin
    @fermions f
    H = hilbert_space(f, 1:4)
    Hsub = hilbert_space(f, [2, 4])
    m = rand(ComplexF64, dim(H), dim(H))

    msub = partial_trace(m, H => Hsub)
    pt = partial_trace(H => Hsub)
    msub_map = pt * reshape(m, (dim(H)^2))
    @test msub ≈ reshape(msub_map, (dim(Hsub), dim(Hsub)))

    H = hilbert_space(f, 1:4, NumberConservation(2))
    Hsub = subregion(hilbert_space(f, [1, 3, 4]), H)
    m = rand(ComplexF64, dim(H), dim(H))
    msub = partial_trace(m, H => Hsub)
    pt = partial_trace(H => Hsub)
    msub_map = pt * reshape(m, (dim(H)^2))
    @test msub ≈ reshape(msub_map, (dim(Hsub), dim(Hsub)))
end

@testitem "Partial trace with missing states" begin
    using LinearAlgebra
    @fermions a
    H1 = hilbert_space(a, 1:2, NumberConservation(1))
    H2 = hilbert_space(a, 3:4, NumberConservation(1))
    H = hilbert_space(a, 1:4, NumberConservation(2))
    m1 = rand(ComplexF64, dim(H1), dim(H1))
    m2 = rand(ComplexF64, dim(H2), dim(H2))
    p = FermionicHilbertSpaces.state_mapper(H, (H1, H2))

    m12 = tensor_product((m1, m2), (H1, H2) => H)
    m1ext = embed(m1, H1 => H)
    m2ext = embed(m2, H2 => H)
    @test m1ext * m2ext ≈ m12
    m21 = tensor_product((m2, m1), (H2, H1) => H)
    @test m2ext * m1ext ≈ m21
    alg1 = FermionicHilbertSpaces.FullPartialTraceAlg()
    alg2 = FermionicHilbertSpaces.SubsystemPartialTraceAlg()
    m12_traced1_alg1 = partial_trace(m12, H => H1; alg=alg1)
    m12_traced2_alg1 = partial_trace(m12, H => H2; alg=alg1)
    m12_traced1_alg2 = partial_trace(m12, H => H1; alg=alg2)
    m12_traced2_alg2 = partial_trace(m12, H => H2; alg=alg2)
    @test m12_traced1_alg1 ≈ m1 * tr(m2)
    @test m12_traced2_alg1 ≈ m2 * tr(m1)
    @test m12_traced1_alg2 ≈ m1 * tr(m2)
    @test m12_traced2_alg2 ≈ m2 * tr(m1)

    m12full = rand(ComplexF64, dim(H), dim(H))
    @test_throws ArgumentError partial_trace(m12full, H => H1; alg=FermionicHilbertSpaces.FullPartialTraceAlg(), skipmissing=false)
    @test partial_trace(m12full, H => H1; alg=FermionicHilbertSpaces.FullPartialTraceAlg(), skipmissing=true) ≈
          partial_trace(m12full, H => H1; alg=FermionicHilbertSpaces.SubsystemPartialTraceAlg()) # This is a bit problematic, because the subsystem partial trace algorithm will not enumerate all possible states in H, and can silently give the wrong result.

    H = hilbert_space(a, 1:4, NumberConservation(1))
    @test_throws ArgumentError generalized_kron((m1, m2), (H1, H2) => H)
    generalized_kron((m1, m2), (H1, H2) => H; skipmissing=true) ≈ tensor_product((m1, m2), (H1, H2) => H) # Since tensor_product first embeds both operators in the full Hilbert space (which can be done) and then multiplies them, it won't complain about missing states.
end