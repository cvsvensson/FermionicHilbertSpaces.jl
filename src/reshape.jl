
function Base.reshape(m::AbstractMatrix, H::AbstractHilbertSpace, Hs)
    splitter = state_splitter(H, Hs)
    _reshape_mat_to_tensor(m, H, Hs, Hs, splitter, splitter)
end
function Base.reshape(m::AbstractMatrix, H::AbstractHilbertSpace, Hsout, Hsin)
    splitterout = state_splitter(H, Hsout)
    splitterin = state_splitter(H, Hsin)
    _reshape_mat_to_tensor(m, H, Hsout, Hsin, splitterout, splitterin)
end
function Base.reshape(m::AbstractVector, H::AbstractHilbertSpace, Hs)
    _reshape_vec_to_tensor(m, H, Hs, state_splitter(H, Hs))
end
const PairWithHilbertSpace = Union{Pair{<:AbstractHilbertSpace,<:Any},Pair{<:Any,<:AbstractHilbertSpace}}
Base.reshape(Hs::PairWithHilbertSpace; kwargs...) = m -> reshape(m, first(Hs), last(Hs); kwargs...)
Base.reshape(m::AbstractArray, Hs::PairWithHilbertSpace; kwargs...) = reshape(m, first(Hs), last(Hs); kwargs...)

function Base.reshape(t::AbstractArray, Hs::Union{<:AbstractVector,Tuple}, H::AbstractHilbertSpace)
    splitter_in = state_splitter(H, Hs)
    if ndims(t) == 2 * length(Hs)
        return _reshape_tensor_to_mat(t, (Hs, splitter_in), (Hs, splitter_in), H, state_splitter)
    elseif ndims(t) == length(Hs)
        return _reshape_tensor_to_vec(t, Hs, H, splitter_in)
    else
        throw(ArgumentError("The number of dimensions in the tensor must match the number of subsystems"))
    end
end

function _reshape_vec_to_tensor(v::AbstractVector, H::AbstractHilbertSpace, Hs, splitter)
    dims = map(dim, Hs)
    fs = basisstates(H)
    Is = map(f -> state_index(f, H), fs)
    t = zeros(eltype(v), dims...)
    for (f, I) in zip(fs, Is)
        for (substates, w) in split_state(f, splitter)
            Iout = state_index.(substates, Hs)
            t[Iout...] += w * v[I]
        end
    end
    return t
end

function _reshape_mat_to_tensor(m::AbstractMatrix, H::AbstractHilbertSpace, Hsout, Hsin, splitterout, splitterin)
    #reshape the matrix m in basis b into a tensor where each index pair has a basis in bs
    dimsin = map(dim, Hsin)
    dimsout = map(dim, Hsout)
    fs = basisstates(H)
    Js = map(f -> state_index(f, H), fs)
    t = zeros(eltype(m), dimsout..., dimsin...)
    for (J1, f1) in zip(Js, fs)
        for (substatesout, wout) in split_state(f1, splitterout)
            Iout = map(state_index, substatesout, Hsout)
            for (J2, f2) in zip(Js, fs)
                for (substatesin, win) in split_state(f2, splitterin)
                    Iin = map(state_index, substatesin, Hsin)
                    t[Iout..., Iin...] += wout * win * m[J1, J2]
                end
            end
        end
    end
    return t
end

function _reshape_tensor_to_mat(t, (Hsout, splitterout), (Hsin, splitterin), H::AbstractHilbertSpace, state_splitter)
    fsout = Base.product(basisstates.(Hsout)...)
    fsin = Base.product(basisstates.(Hsin)...)

    Jouts = map(f -> state_index.(f, Hsout), fsout)
    Jins = map(f -> state_index.(f, Hsin), fsin)

    m = zeros(eltype(t), prod(dim, Hsout), prod(dim, Hsin))
    for (fsin_tuple, Jin) in zip(fsin, Jins)
        for (fullf_in, win) in combine_states(fsin_tuple, splitterin)
            Iin = state_index(fullf_in, H)
            for (fsout_tuple, Jout) in zip(fsout, Jouts)
                for (fullf_out, wout) in combine_states(fsout_tuple, splitterout)
                    Iout = state_index(fullf_out, H)
                    m[Iout, Iin] += wout * win * t[Jout..., Jin...]
                end
            end
        end
    end
    return m
end

function _reshape_tensor_to_vec(t, Hs, H::AbstractHilbertSpace, state_splitter)
    fs = Base.product(basisstates.(Hs)...)
    v = Vector{eltype(t)}(undef, length(fs))
    fill!(v, zero(eltype(v)))
    for fstuple in fs
        Is = state_index.(fstuple, Hs)
        for (fb, w) in combine_states(fstuple, state_splitter)
            Iout = state_index(fb, H)
            ismissing(Iout) && continue
            v[Iout] += w * t[Is...]
        end
    end
    return v
end

@testitem "Reshape" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: project_on_parities, project_on_parity, fermions, majoranas
    function majorana_basis(H)
        b = majoranas(fermions(H))
        labels = collect(keys(b))
        basisops = mapreduce(vec, vcat, [[prod(l -> b[l], ls) for ls in Base.product([labels for _ in 1:n]...) if (issorted(ls) && allunique(ls))] for n in 1:length(labels)])
        pushfirst!(basisops, I + 0 * first(basisops))
        map(Hermitian ∘ (x -> x / sqrt(complex(tr(x * x)))), basisops)
    end

    qns = [NoSymmetry(), ParityConservation(), NumberConservation()]
    for qn in qns
        @fermions f
        H = hilbert_space(f, 1:2, qn)
        majbasis = majorana_basis(H)
        @test all(map(ishermitian, majbasis))
        overlaps = [tr(Γ1' * Γ2) for (Γ1, Γ2) in Base.product(majbasis, majbasis)]
        @test overlaps ≈ I
        @test rank(mapreduce(vec, hcat, majbasis)) == length(majbasis)
    end
    function test_reshape(qn1, qn2, qn3)
        @fermions f
        H1 = hilbert_space(f, [1, 3], qn1)
        H2 = hilbert_space(f, [2, 4], qn2)
        d1 = 4
        d2 = 4
        Hs = (H1, H2)
        H = hilbert_space(f, 1:4, qn3)
        b = fermions(H)
        m = b[1]
        t = reshape(m, H => Hs)
        @test norm(m) ≈ norm(t)

        m = rand(ComplexF64, d1 * d2, d1 * d2)
        t = reshape(m, H => Hs)
        m2 = reshape(t, Hs => H)
        @test m ≈ m2

        v = rand(ComplexF64, d1 * d2)
        tv = reshape(v, H => Hs)
        v2 = reshape(tv, Hs => H)
        @test v ≈ v2
        # Note the how reshaping without phase factors is used in a contraction
        @test sum(reshape(m, H => Hs)[:, :, i, j] * tv[i, j] for i in 1:d1, j in 1:d2) ≈ reshape(m * v, H => Hs)

        m1 = rand(ComplexF64, d1 * d2, d1 * d2)
        m2 = rand(ComplexF64, d1 * d2, d1 * d2)
        t1 = reshape(m1, H => Hs)
        t2 = reshape(m2, H => Hs)
        t3 = zeros(ComplexF64, d1, d2, d1, d2)
        for i in 1:d1, j in 1:d2, k in 1:d1, l in 1:d2, k1 in 1:d1, k2 in 1:d2
            t3[i, j, k, l] += t1[i, j, k1, k2] * t2[k1, k2, k, l]
        end
        @test reshape(m1 * m2, H => Hs) ≈ t3
        @test m1 * m2 ≈ reshape(t3, Hs => H)

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

        Hvirtual3 = [tr(Γ' * h) / sqrt(tr(Γ' * Γ) + 0im) for Γ in basis12all]
        @test Hvirtual3 ≈ Hvirtual
        Hvirtual4 = [tr(Γ' * Hotherbasis) for Γ in basis12normalized]
        @test Hvirtual4 ≈ Hvirtual
        # @test svdvals(Hvirtual) ≈ svdvals(Hvirtual4)

        # t_no_oddodd = reshape(H_no_oddodd, H, Hs)
        # Hvirtual_no_oddodd2 = FermionicHilbertSpaces.reshape_to_matrix(t_no_oddodd, (1, 3))
        # @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd2)
        Hvirtual_no_oddodd3 = [tr(Γ' * H_no_oddodd) / sqrt(tr(Γ' * Γ) + 0im) for Γ in basis12all]
        @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd3)
        Hvirtual_no_oddodd4 = [tr(Γ' * H_no_oddodd_otherbasis) for Γ in basis12normalized]
        @test svdvals(Hvirtual_no_oddodd) ≈ svdvals(Hvirtual_no_oddodd4)

        ## Test consistency with partial trace
        m2 = partial_trace(m, H => H2; phase_factors=false)
        t = reshape(m, H => Hs)
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

        F = partial_trace(m * generalized_kron((m1, I), Hs => H), H => H2)
        @test tr(F * m2) ≈ tr(m * generalized_kron((m1, I), Hs, H) * generalized_kron((I, m2), Hs, H))

        t = reshape(m, H => Hs)
        tpt = sum(t[k1, :, k2, :] * m1[k2, k1] for k1 in axes(t, 1), k2 in axes(t, 3))
        @test partial_trace(m * generalized_kron((m1, I), Hs, H; phase_factors=false), H => H2; phase_factors=false) ≈ tpt

        ## More bases
        H3 = hilbert_space(f, 5:5, qn3)
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
