
"""
    tensor_product(Hs)

Compute the tensor_product product hilbert spaces `Hs`. The symmetry of the resulting basis is computed by promote_symmetry.
"""
tensor_product(Hs::AbstractVector{<:AbstractHilbertSpace}) = foldl(tensor_product, Hs)
tensor_product(Hs::Tuple) = foldl(tensor_product, Hs)

function tensor_product(H1::SymmetricFockHilbertSpace, H2::SymmetricFockHilbertSpace)
    tensor_product_combine_focknumbers(H1, H2)
end

tensor_product(H1::AbstractHilbertSpace, H2::AbstractHilbertSpace) = tensor_product_combine_focknumbers(H1, H2)

function tensor_product_combine_focknumbers(H1, H2)
    isdisjoint(keys(H1.jw), keys(H2.jw)) || throw(ArgumentError("The labels of the two bases are not disjoint"))
    newlabels = vcat(collect(keys(H1.jw)), collect(keys(H2.jw)))
    M1 = length(H1.jw)
    newfocknumbers = vec([f1 + shift_right(f2, M1) for f1 in focknumbers(H1), f2 in focknumbers(H2)])
    FockHilbertSpace(newlabels, newfocknumbers)
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
    @test size(H1) .* size(H2) == size(Hw)

    H1 = SymmetricFockHilbertSpace(1:2, FermionConservation())
    H2 = SymmetricFockHilbertSpace(3:4, FermionConservation())
    Hw = tensor_product(H1, H2)
    H3 = SymmetricFockHilbertSpace(1:4, FermionConservation())
    @test sort(focknumbers(Hw), by=f -> f.f) == sort(focknumbers(H3), by=f -> f.f)
    @test size(H1) .* size(H2) == size(Hw)

    H1 = SymmetricFockHilbertSpace(1:2, ParityConservation())
    H2 = SymmetricFockHilbertSpace(3:4, ParityConservation())
    Hw = tensor_product(H1, H2)
    H3 = SymmetricFockHilbertSpace(1:4, ParityConservation())
    @test sort(focknumbers(Hw), by=f -> f.f) == sort(focknumbers(H3), by=f -> f.f)
    @test size(H1) .* size(H2) == size(Hw)

end


function check_tensor_product_basis_compatibility(b1::AbstractHilbertSpace, b2::AbstractHilbertSpace, b3::AbstractHilbertSpace)
    if vcat(collect(keys(b1)), collect(keys(b2))) != collect(keys(b3))
        throw(ArgumentError("The labels of the output basis are not the same (or ordered the same) as the labels of the input bases. $(keys(b1)) * $(keys(b2)) != $(keys(b3))"))
    end
end

##

"""
    fermionic_kron(ms, Hs, H::AbstractHilbertSpace=tensor_product(Hs))

Compute the fermionic tensor product of matrices or vectors in `ms` with respect to the spaces `Hs`, respectively. Return a matrix in the space `H`, which defaults to the tensor_product product of `Hs`.
"""
function fermionic_kron(ms, Hs, H::AbstractHilbertSpace=tensor_product(Hs), phase_factors::Bool=true)
    N = ndims(first(ms))
    mout = allocate_tensor_product_result(ms, Hs)

    fermionpositions = map(Base.Fix2(siteindices, H.jw) ∘ collect ∘ keys, Hs)
    fockmapper = FockMapper(fermionpositions)
    if N == 1
        return fermionic_kron_vec!(mout, Tuple(ms), Tuple(Hs), H, fockmapper)
    elseif N == 2
        return fermionic_kron_mat!(mout, Tuple(ms), Tuple(Hs), H, fockmapper, phase_factors)
    end
    throw(ArgumentError("Only 1D or 2D arrays are supported"))
end

fermionic_kron(Hs::Pair, phase_factors::Bool=true) = (ms...) -> fermionic_kron(ms, Hs, phase_factors)
fermionic_kron(ms, Hs::Pair, phase_factors::Bool=true) = fermionic_kron(ms, first(Hs), last(Hs), phase_factors)


uniform_to_sparse_type(::Type{UniformScaling{T}}) where {T} = SparseMatrixCSC{T,Int}
uniform_to_sparse_type(::Type{T}) where {T} = T
function allocate_tensor_product_result(ms, bs)
    T = Base.promote_eltype(ms...)
    N = ndims(first(ms))
    types = map(uniform_to_sparse_type ∘ typeof, ms)
    MT = Base.promote_op(kron, types...)
    dimlengths = map(length ∘ focknumbers, bs)
    Nout = prod(dimlengths)
    _mout = Zeros(T, ntuple(j -> Nout, N))
    try
        convert(MT, _mout)
    catch
        Array(_mout)
    end
end

tensor_product_iterator(m, ::AbstractFockHilbertSpace) = findall(!iszero, m)
tensor_product_iterator(::UniformScaling, H::AbstractFockHilbertSpace) = diagind(I(length(focknumbers(H))), IndexCartesian())

function fermionic_kron_mat!(mout, ms::Tuple, Hs::Tuple, H::AbstractFockHilbertSpace, fockmapper, phase_factors::Bool=true)
    fill!(mout, zero(eltype(mout)))
    jw = H.jw
    partition = map(collect ∘ keys, Hs) # using collect here turns out to be a bit faster
    isorderedpartition(partition, jw) || throw(ArgumentError("The partition must be ordered according to jw"))

    inds = Base.product(map(tensor_product_iterator, ms, Hs)...)
    for I in inds
        I1 = map(i -> i[1], I)
        I2 = map(i -> i[2], I)
        fock1 = map(indtofock, I1, Hs)
        fullfock1 = fockmapper(fock1)
        outind1 = focktoind(fullfock1, H)
        fock2 = map(indtofock, I2, Hs)
        fullfock2 = fockmapper(fock2)
        outind2 = focktoind(fullfock2, H)
        s = phase_factors ? phase_factor_h(fullfock1, fullfock2, partition, jw) : 1
        v = mapreduce((m, i1, i2) -> m[i1, i2], *, ms, I1, I2)
        mout[outind1, outind2] += v * s
    end
    return mout
end

function fermionic_kron_vec!(mout, ms::Tuple, Hs::Tuple, H::AbstractFockHilbertSpace, fockmapper)
    fill!(mout, zero(eltype(mout)))
    U = embedding_unitary(Hs, H)
    dimlengths = map(length ∘ focknumbers, Hs)
    inds = CartesianIndices(Tuple(dimlengths))
    for I in inds
        TI = Tuple(I)
        fock = map(indtofock, TI, Hs)
        fullfock = fockmapper(fock)
        outind = focktoind(fullfock, H)
        mout[outind] += mapreduce((i1, m) -> m[i1], *, TI, ms)
    end
    return U * mout
end

"""
    tensor_product(ms, Hs, H::AbstractHilbertSpace, phase_factors=true)

Compute the ordered product of the fermionic embeddings of the matrices `ms` in the spaces `Hs` into the space `H`.
"""
function tensor_product(ms::Union{<:AbstractVector,<:Tuple}, Hs, H::AbstractHilbertSpace, phase_factors::Bool=true)
    # See eq. 26 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isorderedpartition(Hs, H) || throw(ArgumentError("The subsystems must be a partition consistent with the jordan-wigner ordering of the full system"))
    return mapreduce(((m, fine_basis),) -> embedding(m, fine_basis, H, phase_factors), *, zip(ms, Hs))
end
tensor_product(ms::Union{<:AbstractVector,<:Tuple}, HsH::Pair{<:Any,<:AbstractFockHilbertSpace}, phase_factors::Bool=true) = tensor_product(ms, first(HsH), last(HsH), phase_factors)
tensor_product(HsH::Pair{<:Any,<:AbstractFockHilbertSpace}, phase_factors::Bool=true) = (ms...) -> tensor_product(ms, first(HsH), last(HsH), phase_factors)

@testitem "Fermionic tensor product properties" begin
    # Properties from J. Phys. A: Math. Theor. 54 (2021) 393001
    # Eq. 16
    using Random, Base.Iterators, LinearAlgebra
    import FermionicHilbertSpaces: embedding, tensor_product, embedding_unitary, canonical_embedding, project_on_parity, project_on_parities

    Random.seed!(1)
    N = 7
    rough_size = 5
    fine_size = 3
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
    rhs = fermionic_kron(reduce(vcat, ops_fine), reduce(vcat, Hs_fine), H)
    finetensor_products = [fermionic_kron(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, Hs_fine, Hs_rough)]
    lhs = fermionic_kron(finetensor_products, Hs_rough, H)
    @test lhs ≈ rhs

    rhs = kron(reduce(vcat, ops_fine), reduce(vcat, Hs_fine), H)
    lhs = kron([kron(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, Hs_fine, Hs_rough)], Hs_rough, H)
    @test lhs ≈ rhs

    physical_ops_rough = [project_on_parity(op, H, 1) for (op, H) in zip(ops_rough, Hs_rough)]

    # Eq. 18
    As = ops_rough
    Bs = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    lhs = tr(fermionic_kron(As, Hs_rough, H)' * fermionic_kron(Bs, Hs_rough, H))
    rhs = mapreduce((A, B) -> tr(A' * B), *, As, Bs)
    @test lhs ≈ rhs

    # Fermionic embedding

    # Eq. 19 
    As_modes = [rand(ComplexF64, 2, 2) for _ in 1:N]
    ξ = vcat(fine_partitions...)
    ξbases = vcat(Hs_fine...)
    modebases = [hilbert_space(j:j) for j in 1:N]
    lhs = prod(j -> embedding(As_modes[j], modebases[j], H), 1:N)
    rhs_ordered_prod(X, basis) = mapreduce(j -> embedding(As_modes[j], modebases[j], basis), *, X)
    rhs = fermionic_kron([rhs_ordered_prod(X, H) for (X, H) in zip(ξ, ξbases)], ξbases, H)
    @test lhs ≈ rhs

    # Associativity (Eq. 21)
    @test embedding(embedding(ops_fine[1][1], Hs_fine[1][1], Hs_rough[1]), Hs_rough[1], H) ≈ embedding(ops_fine[1][1], Hs_fine[1][1], H)
    @test all(map(Hs_rough, Hs_fine, ops_fine) do cr, cfs, ofs
        all(map(cfs, ofs) do cf, of
            embedding(embedding(of, cf, cr), cr, H) ≈ embedding(of, cf, H)
        end)
    end)

    ## Eq. 22
    HX = Hs_rough[1]
    Ux = embedding_unitary(rough_partitions, H)
    A = ops_rough[1]
    @test Ux !== I
    @test embedding(A, HX, H) ≈ Ux * canonical_embedding(A, HX, H) * Ux'
    # Eq. 93
    @test tensor_product(physical_ops_rough, Hs_rough, H) ≈ Ux * kron(physical_ops_rough, Hs_rough, H) * Ux'

    # Eq. 23
    X = rough_partitions[1]
    HX = Hs_rough[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^length(X), 2^length(X))
    #Eq 5a and 5br are satisfied also when embedding matrices in larger subsystems
    @test embedding(A, HX, H)' ≈ embedding(A', HX, H)
    @test canonical_embedding(A, HX, H) * canonical_embedding(B, HX, H) ≈ canonical_embedding(A * B, HX, H)
    for cmode in modebases
        #Eq 5bl
        local A = rand(ComplexF64, 2, 2)
        local B = rand(ComplexF64, 2, 2)
        @test embedding(A, cmode, H) * embedding(B, cmode, H) ≈ embedding(A * B, cmode, H)
    end

    # Ordered product of embeddings

    # Eq. 31
    A = ops_rough[1]
    X = rough_partitions[1]
    Xbar = setdiff(1:N, X)
    HX = Hs_rough[1]
    HXbar = hilbert_space(Xbar)
    corr = embedding(A, HX, H)
    @test corr ≈ fermionic_kron([A, I], [HX, HXbar], H) ≈ tensor_product([A, I], [HX, HXbar], H) ≈ tensor_product([I, A], [HXbar, HX], H)

    # Eq. 32
    @test tensor_product(As_modes, modebases, H) ≈ fermionic_kron(As_modes, modebases, H)

    ## Fermionic partial trace

    # Eq. 36
    X = rough_partitions[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^N, 2^N)
    HX = Hs_rough[1]
    lhs = tr(embedding(A, HX, H)' * B)
    rhs = tr(A' * partial_trace(B, H, HX))
    @test lhs ≈ rhs

    # Eq. 38 (using A, X, HX, HXbar from above)
    B = rand(ComplexF64, 2^length(Xbar), 2^length(Xbar))
    Hs = [HX, HXbar]
    ops = [A, B]
    @test partial_trace(fermionic_kron(ops, Hs, H), H, HX) ≈ partial_trace(tensor_product(ops, Hs, H), H, HX) ≈ partial_trace(tensor_product(reverse(ops), reverse(Hs), H), H, HX) ≈ A * tr(B)

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
    @test svdvals(Matrix(tensor_product(physical_ops, Hs, H))) ≈ svdvals(Matrix(kron(physical_ops, Hs, H)))
    # However, it is more general. The unitary equivalence holds as long as all except at most one of the operators has a definite parity:

    numberops = map(numberoperator, Hs)
    Uemb = embedding_unitary(Hs, H)
    fine_partition = reduce(vcat, fine_partitions)
    for parities in Base.product([[-1, 1] for _ in 1:length(Hs)]...)
        projected_ops = [project_on_parity(op, H, p) for (op, H, p) in zip(ops, Hs, parities)] # project on local parity
        opsk = [[projected_ops[1:k-1]..., ops[k], projected_ops[k+1:end]...] for k in 1:length(ops)] # switch out one operator of definite parity for an operator of indefinite parity
        embedding_prods = [tensor_product(ops, Hs, H) for ops in opsk]
        kron_prods = [kron(ops, Hs, H) for ops in opsk]

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
    unitaries = [Diagonal([phase(k, f) for f in focknumbers(H)]) * Uemb for k in 1:length(opsk)]
    embedding_prods = [tensor_product(ops, Hs, H) for ops in opsk]
    kron_prods = [kron(ops, Hs, H) for ops in opsk]
    @test all(op1 ≈ U * op2 * U for (op1, op2, U) in zip(embedding_prods, kron_prods, unitaries))

end


@testitem "tensor_product" begin
    using Random, LinearAlgebra
    import SparseArrays: SparseMatrixCSC
    Random.seed!(1234)

    for qn in [NoSymmetry(), ParityConservation(), FermionConservation()]
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
        @test typeof(kron((b1[1], b2[2]), Hs, H3)) == typeof(b1[1])
        @test typeof(tensor_product((b1[1], I), Hs => H3)) == typeof(b1[1])
        @test typeof(kron((b1[1], I), Hs, H3)) == typeof(b1[1])
        @test tensor_product((I, I), Hs => H3) isa SparseMatrixCSC
        @test kron((I, I), Hs, H3) isa SparseMatrixCSC

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
phase_map(N::Int) = phase_map(map(FockNumber, 0:2^N-1), N)
phase_map(H::AbstractFockHilbertSpace) = phase_map(collect(focknumbers(H)), length(H.jw))
LazyPhaseMap(N::Int) = LazyPhaseMap{N,FockNumber{Int}}(map(FockNumber, 0:2^N-1))
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
    @test FermionicHilbertSpaces.fermionic_tensor_product_with_kron_and_maps(ms, (p1, p1), p2) == fermionic_kron(ms, (H1, H2), H12)
end

function fermionic_tensor_product_with_kron_and_maps(ops, phis, phi)
    phi(kron(reverse(map((phi, op) -> phi(op), phis, ops))...))
end


"""
    partial_trace(m::AbstractMatrix,  bHfull::AbstractHilbertSpace, Hsub::AbstractHilbertSpace)

Compute the partial trace of a matrix `m`, leaving the subsystem defined by the basis `bsub`.
"""
function partial_trace(m::AbstractMatrix{T}, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, phase_factors::Bool=true) where {T}
    mout = zeros(T, size(Hsub))
    partial_trace!(mout, m, H, Hsub, phase_factors)
end

partial_trace(Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}, phase_factors::Bool=true) = m -> partial_trace(m, Hs..., phase_factors)
partial_trace(m, Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}, phase_factors::Bool=true) = partial_trace(m, Hs..., phase_factors)

"""
    partial_trace!(mout, m::AbstractMatrix, H::AbstractHilbertSpace, Hout::AbstractHilbertSpace, phase_factors)

Compute the fermionic partial trace of a matrix `m` in basis `H`, leaving only the subsystems specified by `labels`. The result is stored in `mout`, and `Hout` determines the ordering of the basis states.
"""
function partial_trace!(mout, m::AbstractMatrix, H::AbstractHilbertSpace, Hout::AbstractHilbertSpace, phase_factors::Bool=true)
    M = length(H.jw)
    labels = collect(keys(Hout))
    if phase_factors
        consistent_ordering(labels, H.jw) || throw(ArgumentError("Subsystem must be ordered in the same way as the full system"))
    end
    N = length(labels)
    fill!(mout, zero(eltype(mout)))
    subfockstates = focknumbers(Hout)
    Hbar_labels = setdiff(collect(keys(H)), collect(keys(Hout)))
    Hbar = SimpleFockHilbertSpace(Hbar_labels)
    barfockstates = focknumbers(Hbar)
    fm = FockMapper((Hout, Hbar), H)
    for f1 in subfockstates, f2 in subfockstates
        s2 = phase_factors ? phase_factor_f(f1, f2, N) : 1
        I1 = focktoind(f1, Hout)
        I2 = focktoind(f2, Hout)
        for fbar in barfockstates
            fullf1 = fm((f1, fbar))
            fullf2 = fm((f2, fbar))
            s1 = phase_factors ? phase_factor_f(fullf1, fullf2, M) : 1
            s = s2 * s1
            mout[I1, I2] += s * m[focktoind(fullf1, H), focktoind(fullf2, H)]
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
    P = embedding(parityoperator(Hsub), Hsub, H)
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
    op = rand(ComplexF64, size(H))
    local_parity_iter = (1, -1)
    all_parities = Base.product([local_parity_iter for _ in 1:length(Hs)]...)
    @test sum(project_on_parities(op, H, Hs, parities) for parities in all_parities) ≈ op

    ops = [rand(ComplexF64, size(H)) for H in Hs]
    for parities in all_parities
        projected_ops = [project_on_parity(op, Hsub, parity) for (op, Hsub, parity) in zip(ops, Hs, parities)]
        local op = tensor_product(projected_ops, Hs, H)
        @test op ≈ project_on_parities(op, H, Hs, parities)
    end
end
