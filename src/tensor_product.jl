
"""
    tensor_product(Hs)

Return the tensor product space of hilbert spaces `Hs`.
"""
tensor_product(Hs::AbstractVector{<:AbstractHilbertSpace}) = foldl(tensor_product, Hs)
tensor_product(Hs::Tuple) = foldl(tensor_product, Hs)
tensor_product(H1::AbstractHilbertSpace, H2::AbstractHilbertSpace, Hs...) = tensor_product(tensor_product(H1, H2), Hs...)

function tensor_product(H1::SymmetricFockHilbertSpace, H2::SymmetricFockHilbertSpace)
    tensor_product_combine_basisstates(H1, H2)
end
tensor_product(H1::AbstractFockHilbertSpace, H2::AbstractFockHilbertSpace) = tensor_product_combine_basisstates(H1, H2)

tensor_product(H1::AbstractHilbertSpace) = H1
tensor_product(H1::AbstractFockHilbertSpace, H2::AbstractHilbertSpace) = ProductSpace(H1, (H2,))
tensor_product(H1::AbstractHilbertSpace, H2::AbstractFockHilbertSpace) = ProductSpace(H2, (H1,))
tensor_product(H1::AbstractHilbertSpace, H2::AbstractHilbertSpace) = ProductSpace(nothing, (H1, H2))
tensor_product(H::ProductSpace, H2::AbstractHilbertSpace) = ProductSpace(H.fock_space, (H.other_spaces..., H2))
tensor_product(H1::AbstractHilbertSpace, H::ProductSpace) = ProductSpace(H.fock_space, (H1, H.other_spaces...))
tensor_product(H1::AbstractFockHilbertSpace, H::ProductSpace) = ProductSpace(tensor_product(H1, H.fock_space), H.other_spaces)
tensor_product(H::ProductSpace, H2::AbstractFockHilbertSpace) = ProductSpace(tensor_product(H.fock_space, H2), H.other_spaces)
tensor_product(H::AbstractHilbertSpace, ::Nothing) = H
tensor_product(::Nothing, H::AbstractHilbertSpace) = H

function tensor_product_combine_basisstates(H1, H2)
    isdisjoint(keys(H1.jw), keys(H2.jw)) || throw(ArgumentError("The labels of the two bases are not disjoint"))
    newlabels = vcat(collect(keys(H1.jw)), collect(keys(H2.jw)))
    newbasisstates = vec([combine_states(f1, f2, H1, H2) for f1 in basisstates(H1), f2 in basisstates(H2)])
    FockHilbertSpace(newlabels, newbasisstates)
end

function simple_tensor_product(H1::AbstractFockHilbertSpace, H2::AbstractFockHilbertSpace)
    isdisjoint(keys(H1.jw), keys(H2.jw)) || throw(ArgumentError("The labels of the two bases are not disjoint"))
    newlabels = vcat(collect(keys(H1.jw)), collect(keys(H2.jw)))
    SimpleFockHilbertSpace(newlabels)
end
tensor_product(H1::SimpleFockHilbertSpace, H2::SimpleFockHilbertSpace) = simple_tensor_product(H1, H2)

@testitem "tensor_product product of Fock Hilbert Spaces" begin
    using FermionicHilbertSpaces
    H1 = FockHilbertSpace(1:2)
    H2 = FockHilbertSpace(3:4)
    Hw = tensor_product(H1, H2)
    H3 = FockHilbertSpace(1:4)
    @test Hw == H3
    @test dim(H1) * dim(H2) == dim(Hw)

    H1 = SymmetricFockHilbertSpace(1:2, number_conservation())
    H2 = SymmetricFockHilbertSpace(3:4, number_conservation())
    Hw = tensor_product(H1, H2)
    H3 = SymmetricFockHilbertSpace(1:4, number_conservation())
    @test sort(basisstates(Hw)) == sort(basisstates(H3))
    @test dim(H1) * dim(H2) == dim(Hw)

    H1 = SymmetricFockHilbertSpace(1:2, ParityConservation())
    H2 = SymmetricFockHilbertSpace(3:4, ParityConservation())
    Hw = tensor_product(H1, H2)
    H3 = SymmetricFockHilbertSpace(1:4, ParityConservation())
    @test sort(basisstates(Hw)) == sort(basisstates(H3))
    @test dim(H1) * dim(H2) == dim(Hw)

end


function check_tensor_product_basis_compatibility(b1::AbstractHilbertSpace, b2::AbstractHilbertSpace, b3::AbstractHilbertSpace)
    if vcat(collect(keys(b1)), collect(keys(b2))) != collect(keys(b3))
        throw(ArgumentError("The labels of the output basis are not the same (or ordered the same) as the labels of the input bases. $(keys(b1)) * $(keys(b2)) != $(keys(b3))"))
    end
end

size_compatible(m::AbstractMatrix, H) = size(m) == (dim(H), dim(H))
size_compatible(m::UniformScaling, H) = true
size_compatible(m::AbstractVector, H) = length(m) == dim(H)
kron_sizes_compatible(ms, Hs) = all(size_compatible(m, H) for (m, H) in zip(ms, Hs))

@testitem "Size compatible" begin
    using LinearAlgebra, Random
    import FermionicHilbertSpaces: kron_sizes_compatible
    Random.seed!(1234)
    H1 = hilbert_space(1:2)
    H2 = hilbert_space(3:3)
    H3 = hilbert_space(4:6)
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
    mout = allocate_tensor_product_result(ms, Hs)
    extend_state = StateExtender(Hs, H)
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
function allocate_tensor_product_result(ms, bs)
    T = Base.promote_eltype(ms...)
    N = ndims(first(ms))
    types = map(uniform_to_sparse_type ∘ typeof, ms)
    MT = Base.promote_op(kron, types...)
    dimlengths = map(length ∘ basisstates, bs)
    Nout = prod(dimlengths)
    _mout = Zeros(T, ntuple(j -> Nout, N))
    try
        convert(MT, _mout)
    catch
        Array(_mout)
    end
end

tensor_product_iterator(m, ::AbstractHilbertSpace) = findall(!iszero, m)
tensor_product_iterator(::UniformScaling, H::AbstractHilbertSpace) = diagind(I(length(basisstates(H))), IndexCartesian())

function generalized_kron_mat!(mout::AbstractMatrix{T}, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state; phase_factors=true) where T
    fill!(mout, zero(T))
    # ispartition(Hs, H) || throw(ArgumentError("The subsystems must be a partition of the full system"))
    (isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be consistent with the jordan-wigner ordering of the full system")))

    inds = Base.product(map(tensor_product_iterator, ms, Hs)...)
    for I in inds
        I1 = map(i -> i[1], I)
        I2 = map(i -> i[2], I)
        fock1 = map(basisstate, I1, Hs)
        fullfock1 = extend_state(fock1)
        fock2 = map(basisstate, I2, Hs)
        fullfock2 = extend_state(fock2)
        outind1 = state_index(fullfock1, H)
        outind2 = state_index(fullfock2, H)
        s = phase_factors ? phase_factor_h(fullfock1, fullfock2, Hs, H) : 1
        v = one(T)
        for (m, i1, i2) in zip(ms, I1, I2)
            v *= m[i1, i2]
        end
        mout[outind1, outind2] += v * s
    end
    return mout
end

function generalized_kron_vec!(mout, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state; phase_factors=true)
    fill!(mout, zero(eltype(mout)))
    U = embedding_unitary(Hs, H)
    dimlengths = map(length ∘ basisstates, Hs)
    inds = CartesianIndices(Tuple(dimlengths))
    for I in inds
        TI = Tuple(I)
        fock = map(basisstate, TI, Hs)
        fullfock = extend_state(fock)
        outind = state_index(fullfock, H)
        mout[outind] += mapreduce((i1, m) -> m[i1], *, TI, ms)
    end
    return U * mout
end
embedding_unitary(Hs, H::ProductSpace{Nothing}) = I

"""
    tensor_product(ms, Hs, H::AbstractHilbertSpace; kwargs...)

Compute the ordered product of the fermionic embeddings of the matrices `ms` in the spaces `Hs` into the space `H`.
`kwargs` can be passed a bool `phase_factors` and a hilbert space `complement`.
"""
function tensor_product(ms::Union{<:AbstractVector,<:Tuple}, Hs, H::AbstractFockHilbertSpace; kwargs...)
    # See eq. 26 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isorderedpartition(Hs, H) || throw(ArgumentError("The subsystems must be a partition consistent with the jordan-wigner ordering of the full system"))
    return mapreduce(((m, fine_basis),) -> embed(m, fine_basis, H; kwargs...), *, zip(ms, Hs))
end
tensor_product(ms::Union{<:AbstractVector,<:Tuple}, HsH::Pair{<:Any,<:AbstractHilbertSpace}; kwargs...) = tensor_product(ms, first(HsH), last(HsH); kwargs...)
tensor_product(HsH::Pair{<:Any,<:AbstractHilbertSpace}; kwargs...) = (ms...) -> tensor_product(ms, first(HsH), last(HsH); kwargs...)

function tensor_product(ms::Union{<:AbstractVector,<:Tuple}, Hs, H::ProductSpace{Nothing}; kwargs...)
    generalized_kron(ms, Hs, H; kwargs...)
end

@testitem "Fermionic tensor product properties" begin
    # Properties from J. Phys. A: Math. Theor. 54 (2021) 393001
    # Eq. 16
    using Random, Base.Iterators, LinearAlgebra
    import FermionicHilbertSpaces: embedding_unitary, project_on_parity, project_on_parities

    Random.seed!(3)
    N = 8
    rough_size = 4
    fine_size = 2
    rough_partitions = sort.(collect(partition(randperm(N), rough_size)))
    # divide each part of rough partition into finer partitions
    fine_partitions = map(rough_partition -> sort.(collect(partition(shuffle(rough_partition), fine_size))), rough_partitions)
    H = hilbert_space(1:N)
    c = fermions(H)
    Hs_rough = [hilbert_space(r_p) for r_p in rough_partitions]
    Hs_fine = map(f_p_list -> hilbert_space.(f_p_list), fine_partitions)

    ops_rough = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    ops_fine = map(f_p_list -> [rand(ComplexF64, 2^length(f_p), 2^length(f_p)) for f_p in f_p_list], fine_partitions)

    # Associativity (Eq. 16)
    rhs = generalized_kron(reduce(vcat, ops_fine), reduce(vcat, Hs_fine), H)
    finetensor_products = [generalized_kron(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, Hs_fine, Hs_rough)]
    lhs = generalized_kron(finetensor_products, Hs_rough, H)
    @test lhs ≈ rhs

    physical_ops_rough = [project_on_parity(op, H, 1) for (op, H) in zip(ops_rough, Hs_rough)]

    # Eq. 18
    As = ops_rough
    Bs = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    lhs = tr(generalized_kron(As, Hs_rough, H)' * generalized_kron(Bs, Hs_rough, H))
    rhs = mapreduce((A, B) -> tr(A' * B), *, As, Bs)
    @test lhs ≈ rhs

    # Fermionic embedding

    # Eq. 19 
    As_modes = [rand(ComplexF64, 2, 2) for _ in 1:N]
    ξ = vcat(fine_partitions...)
    ξbases = vcat(Hs_fine...)
    modebases = [hilbert_space(j:j) for j in 1:N]
    lhs = prod(j -> embed(As_modes[j], modebases[j], H), 1:N)
    rhs_ordered_prod(X, basis) = mapreduce(j -> Matrix(embed(As_modes[j], modebases[j], basis)), *, X)
    rhs = generalized_kron([rhs_ordered_prod(X, H) for (X, H) in zip(ξ, ξbases)], ξbases, H)
    @test lhs ≈ rhs

    # Associativity (Eq. 21)
    @test embed(embed(ops_fine[1][1], Hs_fine[1][1], Hs_rough[1]), Hs_rough[1], H) ≈ embed(ops_fine[1][1], Hs_fine[1][1], H)
    @test all(map(Hs_rough, Hs_fine, ops_fine) do cr, cfs, ofs
        all(map(cfs, ofs) do cf, of
            embed(embed(of, cf, cr), cr, H) ≈ embed(of, cf, H)
        end)
    end)

    ## Eq. 22
    HX = Hs_rough[1]
    Ux = embedding_unitary(rough_partitions, H)
    A = ops_rough[1]
    @test Ux !== I
    @test embed(A, HX, H) ≈ Ux * embed(A, HX, H; phase_factors=false) * Ux'
    # Eq. 93
    @test tensor_product(physical_ops_rough, Hs_rough, H) ≈ Ux * generalized_kron(physical_ops_rough, Hs_rough, H; phase_factors=false) * Ux'

    # Eq. 23
    X = rough_partitions[1]
    HX = Hs_rough[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^length(X), 2^length(X))
    #Eq 5a and 5br are satisfied also when embedding matrices in larger subsystems
    @test embed(A, HX, H)' ≈ embed(A', HX, H)
    @test embed(A, HX, H; phase_factors=false) * embed(B, HX, H; phase_factors=false) ≈ embed(A * B, HX, H; phase_factors=false)
    for cmode in modebases
        #Eq 5bl
        local A = rand(ComplexF64, 2, 2)
        local B = rand(ComplexF64, 2, 2)
        @test embed(A, cmode, H) * embed(B, cmode, H) ≈ embed(A * B, cmode, H)
    end

    # Ordered product of embeddings

    # Eq. 31
    A = ops_rough[1]
    X = rough_partitions[1]
    Xbar = setdiff(1:N, X)
    HX = Hs_rough[1]
    HXbar = hilbert_space(Xbar)
    corr = embed(A, HX, H)
    @test corr ≈ generalized_kron([A, I], [HX, HXbar], H) ≈ tensor_product([A, I], [HX, HXbar], H) ≈ tensor_product([I, A], [HXbar, HX], H)

    # Eq. 32
    @test tensor_product(As_modes, modebases, H) ≈ generalized_kron(As_modes, modebases, H)

    ## Fermionic partial trace

    # Eq. 36
    X = rough_partitions[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^N, 2^N)
    HX = Hs_rough[1]
    lhs = tr(embed(A, HX, H)' * B)
    rhs = tr(A' * partial_trace(B, H, HX))
    @test lhs ≈ rhs

    # Eq. 38 (using A, X, HX, HXbar from above)
    B = rand(ComplexF64, 2^length(Xbar), 2^length(Xbar))
    Hs = [HX, HXbar]
    ops = [A, B]
    @test partial_trace(generalized_kron(ops, Hs, H), H, HX) ≈ partial_trace(tensor_product(ops, Hs, H), H, HX) ≈ partial_trace(tensor_product(reverse(ops), reverse(Hs), H), H, HX) ≈ A * tr(B)

    # Eq. 39
    A = rand(ComplexF64, 2^N, 2^N)
    X = fine_partitions[1][1]
    Y = rough_partitions[1]
    HX = Hs_fine[1][1]
    HY = Hs_rough[1]
    HZ = H
    Z = 1:N
    rhs = partial_trace(A, HZ, HX)
    lhs = partial_trace(partial_trace(A, HZ, HY), HY, HX)
    @test lhs ≈ rhs

    # Eq. 41
    HY = H
    @test partial_trace(A', HY, HX) ≈ partial_trace(A, HY, HX)'

    # Eq. 95
    ξ = rough_partitions
    Asphys = physical_ops_rough
    Bs = map(X -> rand(ComplexF64, 2^length(X), 2^length(X)), ξ)
    Bsphys = [project_on_parity(B, H, 1) for (B, H) in zip(Bs, Hs_rough)]
    lhs1 = tensor_product(Asphys, Hs_rough, H) * tensor_product(Bsphys, Hs_rough, H)
    rhs1 = tensor_product(Asphys .* Bsphys, Hs_rough, H)
    @test lhs1 ≈ rhs1
    @test tensor_product(Asphys, Hs_rough, H)' ≈ tensor_product(adjoint.(Asphys), Hs_rough, H)

    ## Unitary equivalence between tensor_product and kron
    ops = reduce(vcat, ops_fine)
    Hs = reduce(vcat, Hs_fine)
    physical_ops = [project_on_parity(op, H, 1) for (op, H) in zip(ops, Hs)]
    # Eq. 93 implies that the unitary equivalence holds for the physical operators
    @test svdvals(Matrix(tensor_product(physical_ops, Hs, H))) ≈ svdvals(Matrix(generalized_kron(physical_ops, Hs, H; phase_factors=false)))
    # However, it is more general. The unitary equivalence holds as long as all except at most one of the operators has a definite parity:

    numberops = map(numberoperator, Hs)
    Uemb = embedding_unitary(Hs, H)
    fine_partition = reduce(vcat, fine_partitions)
    for parities in Base.product([[-1, 1] for _ in 1:length(Hs)]...)
        projected_ops = [project_on_parity(op, H, p) for (op, H, p) in zip(ops, Hs, parities)] # project on local parity
        opsk = [[projected_ops[1:k-1]..., ops[k], projected_ops[k+1:end]...] for k in 1:length(ops)] # switch out one operator of definite parity for an operator of indefinite parity
        embedding_prods = [tensor_product(ops, Hs, H) for ops in opsk]
        kron_prods = [generalized_kron(ops, Hs, H; phase_factors=false) for ops in opsk]

        @test all(svdvals(Matrix(op1)) ≈ svdvals(Matrix(op2)) for (op1, op2) in zip(embedding_prods, kron_prods))
    end

    # Explicit construction of unitary equivalence in case of all even (except one) 
    function phase(k, f)
        Xkmask = FermionicHilbertSpaces.focknbr_from_site_labels(fine_partition[k], H.jw)
        iseven(count_ones(f & Xkmask)) && return 1
        phase = 1
        for r in 1:k-1
            Xrmask = FermionicHilbertSpaces.focknbr_from_site_labels(fine_partition[r], H.jw)
            phase *= (-1)^(count_ones(f & Xrmask))
        end
        return phase
    end
    opsk = [[physical_ops[1:k-1]..., ops[k], physical_ops[k+1:end]...] for k in 1:length(ops)]
    unitaries = [Diagonal([phase(k, f) for f in basisstates(H)]) * Uemb for k in 1:length(opsk)]
    embedding_prods = [tensor_product(ops, Hs, H) for ops in opsk]
    kron_prods = [generalized_kron(ops, Hs, H; phase_factors=false) for ops in opsk]
    @test all(op1 ≈ U * op2 * U for (op1, op2, U) in zip(embedding_prods, kron_prods, unitaries))

end


@testitem "tensor_product" begin
    using Random, LinearAlgebra
    import SparseArrays: SparseMatrixCSC
    Random.seed!(1234)

    for qn in [NoSymmetry(), ParityConservation(), number_conservation()]
        H1 = hilbert_space(1:1, qn)
        H2 = hilbert_space(1:3, qn)
        @test_throws ArgumentError tensor_product(H1, H2)
        H2 = hilbert_space(2:3, qn)
        H3 = hilbert_space(1:3, qn)
        H3w = tensor_product(H1, H2)
        @test H3w == tensor_product((H1, H2)) == tensor_product([H1, H2])
        Hs = [H1, H2]
        b1 = fermions(H1)
        b2 = fermions(H2)
        b3 = fermions(H3w)

        #test that they keep sparsity
        @test typeof(tensor_product((b1[1], b2[2]), Hs => H3)) == typeof(b1[1])
        @test typeof(generalized_kron((b1[1], b2[2]), Hs, H3)) == typeof(b1[1])
        @test typeof(tensor_product((b1[1], I), Hs => H3)) == typeof(b1[1])
        @test typeof(generalized_kron((b1[1], I), Hs, H3)) == typeof(b1[1])
        @test tensor_product((I, I), Hs => H3) isa SparseMatrixCSC
        @test generalized_kron((I, I), Hs, H3) isa SparseMatrixCSC

        # Test zero-mode tensor_product
        H1 = hilbert_space(1:0, qn)
        H2 = hilbert_space(1:1, qn)
        c1 = fermions(H1)
        c2 = fermions(H2)
        @test tensor_product([I, I], [H1, H2], H2) == I
        @test tensor_product([I, c2[1]], [H1, H2], H2) == c2[1]
    end

    #Test basis compatibility
    H1 = hilbert_space(1:2, ParityConservation())
    H2 = hilbert_space(2:4, ParityConservation())
    @test_throws ArgumentError tensor_product(H1, H2)
end


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
phase_map(H::AbstractFockHilbertSpace) = phase_map(collect(basisstates(H)), length(H.jw))
LazyPhaseMap(N::Int) = (states = map(FockNumber, UnitRange{UInt64}(0, 2^N - 1)); LazyPhaseMap{N,eltype(states)}(states))
SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(::LazyPhaseMap, rest...) = SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(rest...)
(p::PhaseMap)(op::AbstractMatrix) = p.phases .* op
(p::LazyPhaseMap)(op::AbstractMatrix) = p .* op
@testitem "phasemap" begin
    using LinearAlgebra
    # see App 2 in https://arxiv.org/pdf/2006.03087
    ns = 1:4
    phis = Dict(zip(ns, FermionicHilbertSpaces.phase_map.(ns)))
    lazyphis = Dict(zip(ns, FermionicHilbertSpaces.LazyPhaseMap.(ns)))
    @test all(sum(phis[n].phases .== -1) == (2^n - 2) * 2^n / 2 for n in ns)
    @test all(sum(phis[n].phases .== -1) == (2^n - 2) * 2^n / 2 for n in ns)

    for N in ns
        H = SimpleFockHilbertSpace(1:N)
        c = fermions(H)
        q = FermionicHilbertSpaces.QubitOperators(H)
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

    H1 = hilbert_space(1:1)
    c1 = fermions(H1)
    H2 = hilbert_space(2:2)
    c2 = fermions(H2)
    H12 = hilbert_space(1:2)
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


"""
    partial_trace(m::AbstractMatrix,  Hfull::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; phase_factors = true, complement)

Compute the partial trace of a matrix `m`, leaving the subsystem defined by the basis `Hsub`.
"""
function partial_trace(m::AbstractMatrix{T}, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; phase_factors=use_phase_factors(H) && use_phase_factors(Hsub), complement=complementary_subsystem(H, Hsub)) where {T}
    size_compatible(m, H) || throw(ArgumentError("The size of `m` must match the size of `H`"))
    if H == Hsub
        return copy(m)
    end
    mout = zeros(T, dim(Hsub), dim(Hsub))
    partial_trace!(mout, m, H, Hsub, phase_factors, complement)
end

partial_trace(Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}; kwargs...) = m -> partial_trace(m, Hs...; kwargs...)
partial_trace(m, Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}; kwargs...) = partial_trace(m, Hs...; kwargs...)
use_phase_factors(H::AbstractHilbertSpace) = false
use_phase_factors(H::AbstractFockHilbertSpace) = true

"""
    partial_trace!(mout, m::AbstractMatrix, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, phase_factors, complement)

Compute the fermionic partial trace of a matrix `m` in basis `H`, leaving only the subsystems specified by `labels`. The result is stored in `mout`, and `Hsub` determines the ordering of the basis states.
"""
function partial_trace!(mout, m::AbstractMatrix, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, phase_factors::Bool, complement, extend_state=StateExtender((Hsub, complement), H))
    if phase_factors
        # labels = collect(keys(Hsub))
        # consistent_ordering(labels, mode_ordering(H)) || throw(ArgumentError("Subsystem must be ordered in the same way as the full system"))
        consistent_ordering(Hsub, H) || throw(ArgumentError("Subsystem must be ordered in the same way as the full system"))
    end
    fill!(mout, zero(eltype(mout)))
    substates = basisstates(Hsub)
    barstates = basisstates(complement)
    for f1 in substates, f2 in substates
        s2 = phase_factors ? phase_factor_f(f1, f2, length(keys(Hsub))) : 1
        I1 = state_index(f1, Hsub)
        I2 = state_index(f2, Hsub)
        for fbar in barstates
            fullf1 = extend_state((f1, fbar))
            fullf2 = extend_state((f2, fbar))
            s1 = phase_factors ? phase_factor_f(fullf1, fullf2, length(keys(H))) : 1
            s = s2 * s1
            J1 = state_index(fullf1, H)
            ismissing(J1) && continue
            J2 = state_index(fullf2, H)
            ismissing(J2) && continue
            mout[I1, I2] += s * m[J1, J2]
        end
    end
    return mout
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
    Hs = [hilbert_space(2k-1:2k) for k in 1:3]
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
