
function Base.reshape(m::AbstractMatrix, H::AbstractHilbertSpace, Hs; phase_factors=true)
    _reshape_mat_to_tensor(m, H, Hs, StateSplitter(H, Hs), phase_factors)
end
function Base.reshape(m::AbstractVector, H::AbstractHilbertSpace, Hs; phase_factors=true)
    _reshape_vec_to_tensor(m, H, Hs, StateSplitter(H, Hs), phase_factors)
end
const PairWithHilbertSpace = Union{Pair{<:AbstractHilbertSpace,<:Any},Pair{<:Any,<:AbstractHilbertSpace}}
Base.reshape(Hs::PairWithHilbertSpace; kwargs...) = m -> reshape(m, first(Hs), last(Hs); kwargs...)
Base.reshape(m::AbstractArray, Hs::PairWithHilbertSpace; kwargs...) = reshape(m, first(Hs), last(Hs); kwargs...)

function Base.reshape(t::AbstractArray, Hs::Union{<:AbstractVector,Tuple}, H::AbstractHilbertSpace; phase_factors=true)
    if ndims(t) == 2 * length(Hs)
        return _reshape_tensor_to_mat(t, Hs, H, StateExtender(Hs, H), phase_factors)
    elseif ndims(t) == length(Hs)
        return _reshape_tensor_to_vec(t, Hs, H, StateExtender(Hs, H), phase_factors)
    else
        throw(ArgumentError("The number of dimensions in the tensor must match the number of subsystems"))
    end
end

function _reshape_vec_to_tensor(v::AbstractVector, H::AbstractHilbertSpace, Hs, statesplitter, phase_factors)
    if phase_factors
        isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    dims = length.(basisstates.(Hs))
    fs = basisstates(H)
    Is = map(f -> state_index(f, H), fs)
    Iouts = map(f -> state_index.(statesplitter(f), Hs), fs)
    t = Array{eltype(v),length(Hs)}(undef, dims...)
    for (I, Iout) in zip(Is, Iouts)
        t[Iout...] = v[I...]
    end
    return t
end

function _reshape_mat_to_tensor(m::AbstractMatrix, H::AbstractHilbertSpace, Hs, statesplitter, phase_factors)
    #reshape the matrix m in basis b into a tensor where each index pair has a basis in bs
    if phase_factors
        isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    dims = length.(basisstates.(Hs))
    fs = basisstates(H)
    Is = map(f -> state_index(f, H), fs)
    Iouts = map(f -> state_index.(statesplitter(f), Hs), fs)
    t = Array{eltype(m),2 * length(Hs)}(undef, dims..., dims...)
    partition = map(collect ∘ keys, Hs)
    for (I1, Iout1, f1) in zip(Is, Iouts, fs)
        for (I2, Iout2, f2) in zip(Is, Iouts, fs)
            s = phase_factors ? phase_factor_h(f1, f2, partition, H.jw) : 1
            t[Iout1..., Iout2...] = m[I1, I2] * s
        end
    end
    return t
end

function _reshape_tensor_to_mat(t, Hs, H::AbstractHilbertSpace, stateextender, phase_factors)
    if phase_factors
        isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be ordered according to jw"))
    end
    fs = Base.product(basisstates.(Hs)...)
    fsb = map(stateextender, fs)
    Is = map(f -> state_index.(f, Hs), fs)
    Iouts = map(f -> state_index(f, H), fsb)
    m = Matrix{eltype(t)}(undef, length(fsb), length(fsb))
    partition = map(collect ∘ keys, Hs)

    for (I1, Iout1, f1) in zip(Is, Iouts, fsb)
        for (I2, Iout2, f2) in zip(Is, Iouts, fsb)
            s = phase_factors ? phase_factor_h(f1, f2, partition, H.jw) : 1
            m[Iout1, Iout2] = t[I1..., I2...] * s
        end
    end
    return m
end

function _reshape_tensor_to_vec(t, Hs, H::AbstractHilbertSpace, stateextender, phase_factors)
    isorderedpartition(Hs, H) || throw(ArgumentError("The partition must be ordered according to jw"))
    fs = Base.product(basisstates.(Hs)...)
    v = Vector{eltype(t)}(undef, length(fs))
    for fs in fs
        Is = state_index.(fs, Hs)
        fb = stateextender(fs)
        Iout = state_index(fb, H)
        v[Iout] = t[Is...]
    end
    return v
end

@testitem "Reshape" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: project_on_parities, project_on_parity
    function majorana_basis(H)
        b = fermions(H)
        majoranas = Dict((l, s) => (s == :- ? 1im : 1) * b[l] + hc for (l, s) in Base.product(keys(b), [:+, :-]))
        labels = collect(keys(majoranas))
        basisops = mapreduce(vec, vcat, [[prod(l -> majoranas[l], ls) for ls in Base.product([labels for _ in 1:n]...) if (issorted(ls) && allunique(ls))] for n in 1:length(labels)])
        pushfirst!(basisops, I + 0 * first(basisops))
        map(Hermitian ∘ (x -> x / sqrt(complex(tr(x * x)))), basisops)
    end

    qns = [NoSymmetry(), ParityConservation(), NumberConservation()]
    for qn in qns
        H = hilbert_space(1:2, qn)
        majbasis = majorana_basis(H)
        @test all(map(ishermitian, majbasis))
        overlaps = [tr(Γ1' * Γ2) for (Γ1, Γ2) in Base.product(majbasis, majbasis)]
        @test overlaps ≈ I
        @test rank(mapreduce(vec, hcat, majbasis)) == length(majbasis)
    end

    function test_reshape(qn1, qn2, qn3)
        H1 = hilbert_space((1, 3), qn1)
        H2 = hilbert_space((2, 4), qn2)
        d1 = 4
        d2 = 4
        Hs = (H1, H2)
        H = hilbert_space(sort(vcat(keys(H1)..., keys(H2)...)), qn3)
        b = fermions(H)
        b1 = fermions(H1)
        b2 = fermions(H2)
        m = b[1]
        t = reshape(m, H => Hs)
        m12 = FermionicHilbertSpaces.reshape_to_matrix(t, (1, 3))
        @test rank(m12) == 1
        @test abs(dot(reshape(svd(m12).U, d1, d1, d2^2)[:, :, 1], b1[1])) ≈ norm(b1[1])

        m = b[1] + b[2]
        t = reshape(m, H => Hs)
        m12 = FermionicHilbertSpaces.reshape_to_matrix(t, (1, 3))
        @test rank(m12) == 2

        m = rand(ComplexF64, d1 * d2, d1 * d2)
        t = reshape(m, H => Hs)
        m2 = reshape(t, Hs => H)
        @test m ≈ m2
        t = reshape(m, H => Hs; phase_factors=false) #without phase factors (standard decomposition)
        m2 = reshape(t, Hs => H; phase_factors=false)
        @test m ≈ m2

        v = rand(ComplexF64, d1 * d2)
        tv = reshape(v, H => Hs)
        v2 = reshape(tv, Hs => H)
        @test v ≈ v2
        # Note the how reshaping without phase factors is used in a contraction
        @test sum(reshape(m, H => Hs; phase_factors=false)[:, :, i, j] * tv[i, j] for i in 1:d1, j in 1:d2) ≈ reshape(m * v, H => Hs)

        m1 = rand(ComplexF64, d1 * d2, d1 * d2)
        m2 = rand(ComplexF64, d1 * d2, d1 * d2)
        t1 = reshape(m1, H => Hs; phase_factors=false)
        t2 = reshape(m2, H => Hs; phase_factors=false)
        t3 = zeros(ComplexF64, d1, d2, d1, d2)
        for i in 1:d1, j in 1:d2, k in 1:d1, l in 1:d2, k1 in 1:d1, k2 in 1:d2
            t3[i, j, k, l] += t1[i, j, k1, k2] * t2[k1, k2, k, l]
        end
        @test reshape(m1 * m2, H => Hs; phase_factors=false) ≈ t3
        @test m1 * m2 ≈ reshape(t3, Hs => H; phase_factors=false)

        basis1 = majorana_basis(H1)
        basis2 = majorana_basis(H2)
        basis12all = [generalized_kron((Γ1, Γ2), Hs => H) for (Γ1, Γ2) in Base.product(basis1, basis2)]
        basis12oddodd = [project_on_parities(Γ, H, Hs, (-1, -1)) for Γ in basis12all]
        basis12oddeven = [project_on_parities(Γ, H, Hs, (-1, 1)) for Γ in basis12all]
        basis12evenodd = [project_on_parities(Γ, H, Hs, (1, -1)) for Γ in basis12all]
        basis12eveneven = [project_on_parities(Γ, H, Hs, (1, 1)) for Γ in basis12all]
        basis12normalized = map(x -> x / sqrt(tr(x^2) + 0im), basis12all)

        @test all(map(tr, map(adjoint, basis12all) .* basis12all) .≈ 1)
        overlaps = [tr(Γ1' * Γ2) for (Γ1, Γ2) in Base.product(vec(basis12all), vec(basis12all))]
        @test overlaps ≈ I
        @test all(ishermitian, basis12normalized)
        @test all(map(tr, basis12normalized .* basis12normalized) .≈ 1)
        @test all(filter(x -> abs(x) > 0.01, map(tr, basis12oddodd .* basis12oddodd)) .≈ -1)
        @test all(filter(x -> abs(x) > 0.01, map(tr, basis12eveneven .* basis12eveneven)) .≈ 1)
        @test all(filter(x -> abs(x) > 0.01, map(tr, basis12evenodd .* basis12evenodd)) .≈ 1)
        @test all(filter(x -> abs(x) > 0.01, map(tr, basis12oddeven .* basis12oddeven)) .≈ 1)

        Hvirtual = rand(ComplexF64, length(basis1), length(basis2))
        Hoddoddvirtual = [Hvirtual[I] * norm(basis12oddodd[I]) for I in CartesianIndices(Hvirtual)]
        Hvirtual_no_oddodd = Hvirtual - Hoddoddvirtual
        h = sum(Hvirtual[I] * basis12all[I] for I in CartesianIndices(Hvirtual))
        H_no_oddodd = sum(Hvirtual_no_oddodd[I] * basis12all[I] for I in CartesianIndices(Hvirtual))
        Hotherbasis = sum(Hvirtual[I] * basis12normalized[I] for I in CartesianIndices(Hvirtual))
        H_no_oddodd_otherbasis = sum(Hvirtual_no_oddodd[I] * basis12normalized[I] for I in CartesianIndices(Hvirtual))
        @test H_no_oddodd_otherbasis ≈ H_no_oddodd

        t = reshape(h, H => Hs)
        Hvirtual2 = FermionicHilbertSpaces.reshape_to_matrix(t, (1, 3))
        @test svdvals(Hvirtual) ≈ svdvals(Hvirtual2)
        Hvirtual3 = [tr(Γ' * h) / sqrt(tr(Γ' * Γ) + 0im) for Γ in basis12all]
        @test Hvirtual3 ≈ Hvirtual
        Hvirtual4 = [tr(Γ' * Hotherbasis) for Γ in basis12normalized]
        @test Hvirtual4 ≈ Hvirtual
        # @test svdvals(Hvirtual) ≈ svdvals(Hvirtual4)

        t_no_oddodd = reshape(H_no_oddodd, H, Hs; phase_factors=true)
        Hvirtual_no_oddodd2 = FermionicHilbertSpaces.reshape_to_matrix(t_no_oddodd, (1, 3))
        @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd2)
        Hvirtual_no_oddodd3 = [tr(Γ' * H_no_oddodd) / sqrt(tr(Γ' * Γ) + 0im) for Γ in basis12all]
        @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd3)
        Hvirtual_no_oddodd4 = [tr(Γ' * H_no_oddodd_otherbasis) for Γ in basis12normalized]
        @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd4)

        ## Test consistency with partial trace
        m = rand(ComplexF64, d1 * d2, d1 * d2)
        m2 = partial_trace(m, H => H2; phase_factors=true)
        t = reshape(m, H => Hs; phase_factors=true)
        tpt = sum(t[k, :, k, :] for k in axes(t, 1))
        @test m2 ≈ tpt

        m2 = partial_trace(m, H => H2; phase_factors=false)
        t = reshape(m, H => Hs; phase_factors=false)
        tpt = sum(t[k, :, k, :] for k in axes(t, 1))
        @test m2 ≈ tpt

        mE = project_on_parity(m, H, 1)
        mO = project_on_parity(m, H, -1)
        m1 = rand(ComplexF64, d1, d1)
        m2 = rand(ComplexF64, d2, d2)
        m2O = project_on_parity(m2, H2, -1)
        m2E = project_on_parity(m2, H2, 1)
        m1O = project_on_parity(m1, H1, -1)
        m1E = project_on_parity(m1, H1, 1)
        mEE = project_on_parities(m, H, Hs, (1, 1))
        mOO = project_on_parities(m, H, Hs, (-1, -1))

        F = partial_trace(H => H2)(m * generalized_kron(Hs => H)(m1, I))
        @test tr(F * m2) ≈ tr(m * generalized_kron((m1, I), Hs, H) * generalized_kron((I, m2), Hs, H))

        t = reshape(m, H => Hs; phase_factors=false)
        tpt = sum(t[k1, :, k2, :] * m1[k2, k1] for k1 in axes(t, 1), k2 in axes(t, 3))
        @test partial_trace(m * kron((m1, I), Hs, H), H => H2; phase_factors=false) ≈ tpt

        ## More bases
        H3 = hilbert_space(5:5, qn3)
        d3 = 2
        Hs = (H1, H2, H3)
        H = tensor_product(Hs)
        m = rand(ComplexF64, d1 * d2 * d3, d1 * d2 * d3)
        t = reshape(m, H => Hs)
        @test ndims(t) == 6
        @test m ≈ reshape(t, Hs, H)
    end

    qns_iterator = [[NoSymmetry(), NoSymmetry(), NoSymmetry()],
        [ParityConservation(), ParityConservation(), ParityConservation()],
        [NumberConservation(), NumberConservation(), NumberConservation()],
        [NoSymmetry(), ParityConservation(), NumberConservation()],
        [NumberConservation(), NumberConservation(), NoSymmetry()],
        [ParityConservation(), ParityConservation(), NumberConservation()]]
    for qns in qns_iterator
        test_reshape(qns...)
    end
end

function reshape_to_matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}) where {N,NL}
    rightindices::NTuple{N - NL,Int} = Tuple(setdiff(ntuple(identity, N), leftindices))
    reshape_to_matrix(t, leftindices, rightindices)
end
function reshape_to_matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}, rightindices::NTuple{NR,Int}) where {N,NL,NR}
    @assert NL + NR == N
    tperm = permutedims(t, (leftindices..., rightindices...))
    lsize = prod(i -> size(t, i), leftindices, init=1)
    rsize = prod(i -> size(t, i), rightindices, init=1)
    reshape(tperm, lsize, rsize)
end
