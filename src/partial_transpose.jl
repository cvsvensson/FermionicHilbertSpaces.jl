
partial_transpose_phase_factor(f1, f2, ::AbstractAtomicHilbertSpace) = 1

# _nbr_differing_modes(f1, f2, N::Int) = count(i -> _bit(f1, i) ⊻ _bit(f2, i), 1:N) # generic fallback

# `complement === nothing` means Hsub spans all of H, so states split into (a,) instead of (a, b)
_pt_parts(Hsub, complement) = isnothing(complement) ? (Hsub,) : (Hsub, complement)

_split_all_states(H, mapper) = map(Base.Fix2(split_state, mapper), basisstates(H))
_partial_transpose_out_mapper(H, Hout, Hsub, complement, mapper_in) =
    Hout === H ? mapper_in : state_mapper(Hout, _pt_parts(Hsub, complement))

function _throw_missing_pt_state(g, Hout)
    throw(ArgumentError("The state $g is not in the output Hilbert space. The partial transpose does not preserve number/parity constraints: pass a larger output space via `Hout` (e.g. the unconstrained space on the same modes), or `skipmissing=true` to drop such terms."))
end

function _partial_transpose_eltype(H, Hsub, Hout, mapper_in, mapper_out, phase_factors)
    f = first(basisstates(H))
    splits, ws = split_state(f, mapper_in)
    fsub, fbar... = first(splits)
    _, us = combine_states((fsub, fbar...), mapper_out)
    fout = first(basisstates(Hout))
    sT = phase_factors ? promote_type(typeof(partial_trace_phase_factor(f, f, H)),
        typeof(partial_transpose_phase_factor(fsub, fsub, Hsub)),
        typeof(partial_trace_phase_factor(fout, fout, Hout))) : Int
    return promote_type(sT, eltype(ws), eltype(us))
end

"""
    _foreach_partial_transpose_term(op, inds, H, Hsub, Hout, splits, mapper_out; phase_factors, skipmissing)

Kernel shared by the direct and the sparse-map implementation. For every index pair
`(J1, J2)` in `inds` (matrix element `|f1⟩⟨f2|` of `H`), split `f1 → (a1, b1)`, `f2 → (a2, b2)`,
swap the transposed parts, `|a1 b1⟩⟨a2 b2| ↦ |a2 b1⟩⟨a1 b2|`, recombine into `Hout` and call
`op(J1, J2, K1, K2, v)` with the full coefficient `v`.
"""
function _foreach_partial_transpose_term(op, inds, H, Hsub, Hout, splits, mapper_out; phase_factors=true, skipmissing=false)
    for I in inds
        J1, J2 = I[1], I[2]
        f1 = basisstate(J1, H)
        f2 = basisstate(J2, H)
        s_in = phase_factors ? partial_trace_phase_factor(f1, f2, H) : 1 # Fock(H) → operator basis
        splits1, amps1 = splits[J1]
        splits2, amps2 = splits[J2]
        for ((a1, b1...), w1) in zip(splits1, amps1), ((a2, b2...), w2) in zip(splits2, amps2)
            s_sub = phase_factors ? partial_transpose_phase_factor(a1, a2, Hsub) : 1 # local transpose
            gs1, us1 = combine_states((a2, b1...), mapper_out)
            gs2, us2 = combine_states((a1, b2...), mapper_out)
            for (g1, u1) in zip(gs1, us1)
                K1 = state_index(g1, Hout)
                if iszero(K1)
                    skipmissing && continue
                    _throw_missing_pt_state(g1, Hout)
                end
                for (g2, u2) in zip(gs2, us2)
                    K2 = state_index(g2, Hout)
                    if iszero(K2)
                        skipmissing && continue
                        _throw_missing_pt_state(g2, Hout)
                    end
                    s_out = phase_factors ? partial_trace_phase_factor(g1, g2, Hout) : 1 # operator basis → Fock(Hout)
                    op(J1, J2, K1, K2, s_in * s_sub * s_out * w1 * conj(w2) * u1 * conj(u2))
                end
            end
        end
    end
    return nothing
end

## Direct (in-place) implementation

"""
    partial_transpose!(mout, m, H, Hsub; Hout=H, complement=complementary_subsystem(H, Hsub), phase_factors=true, skipmissing=false)

In-place partial transpose of `m` (operator on `H`) on the subsystem `Hsub`, written to `mout`
(operator on `Hout`). See [`partial_transpose`](@ref).
"""
function partial_transpose!(mout::AbstractMatrix, m::AbstractMatrix, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace,
    complement, Hout::AbstractHilbertSpace,
    mapper_in=state_mapper(H, _pt_parts(Hsub, complement)),
    mapper_out=_partial_transpose_out_mapper(H, Hout, Hsub, complement, mapper_in);
    phase_factors=true, skipmissing=false)
    size(m) == (dim(H), dim(H)) || throw(DimensionMismatch("The input matrix must have size ($(dim(H)), $(dim(H))), got $(size(m))"))
    size(mout) == (dim(Hout), dim(Hout)) || throw(DimensionMismatch("The output matrix must have size ($(dim(Hout)), $(dim(Hout))), got $(size(mout))"))
    fill!(mout, zero(eltype(mout)))
    splits = _split_all_states(H, mapper_in)
    _foreach_partial_transpose_term(tensor_product_iterator(m, H), H, Hsub, Hout, splits, mapper_out; phase_factors, skipmissing) do J1, J2, K1, K2, v
        mout[K1, K2] += v * m[J1, J2]
    end
    return mout
end
function partial_transpose!(mout, m, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), Hout=H, kwargs...)
    partial_transpose!(mout, m, H, Hsub, complement, Hout; kwargs...)
end

"""
    partial_transpose(m, H, Hsub; Hout=H, complement=complementary_subsystem(H, Hsub), phase_factors=true, skipmissing=false)
    partial_transpose(m, H => Hout, Hsub; kwargs...)
    partial_transpose(H, Hsub; kwargs...)  # -> PartialTransposeMap

Partial transpose of `m` (an operator on `H`, a pure state, or a vectorized density matrix)
on the subsystem `Hsub`, which may be any subset of the modes of `H` (contiguous or not).

For fermions this is the fermionic partial transpose (partial time reversal)
`R_A(Ẽ^{ν,ν'}) = i^{k_A} Ẽ^{(ν'_A,ν_B),(ν_A,ν'_B)}` in the fermionic operator basis, which is
independent of the mode ordering. With `phase_factors=false`, or for spaces without phase
factors, it is the ordinary partial transpose.

The partial transpose does not preserve number or parity constraints. For constrained `H`,
pass a larger output space on the same modes as `Hout`. The result is not Hermitian in
general: `(ρ^{R_A})' == p_A ρ^{R_A} p_A`, with `p_A` the parity of `Hsub`.
"""
function partial_transpose(m::AbstractMatrix, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace;
    complement=complementary_subsystem(H, Hsub), Hout=H, phase_factors=true, skipmissing=false)
    mapper_in = state_mapper(H, _pt_parts(Hsub, complement))
    mapper_out = _partial_transpose_out_mapper(H, Hout, Hsub, complement, mapper_in)
    T = promote_type(eltype(m), _partial_transpose_eltype(H, Hsub, Hout, mapper_in, mapper_out, phase_factors))
    mout = zeros(T, dim(Hout), dim(Hout))
    partial_transpose!(mout, m, H, Hsub, complement, Hout, mapper_in, mapper_out; phase_factors, skipmissing)
end

partial_transpose(m, Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}, Hsub::AbstractHilbertSpace; kwargs...) =
    partial_transpose(m, first(Hs), Hsub; Hout=last(Hs), kwargs...)
partial_transpose(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; kwargs...) = PartialTransposeMap(H, Hsub; kwargs...)


# Pure state: direct kernel, no superoperator, and no ψψ† either (iterate nonzero pairs).
function partial_transpose(ψ::AbstractVector, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace;
    complement=complementary_subsystem(H, Hsub), Hout=H, phase_factors=true, skipmissing=false)
    if length(ψ) == dim(H)
        mapper_in = state_mapper(H, _pt_parts(Hsub, complement))
        mapper_out = _partial_transpose_out_mapper(H, Hout, Hsub, complement, mapper_in)
        T = promote_type(eltype(ψ), _partial_transpose_eltype(H, Hsub, Hout, mapper_in, mapper_out, phase_factors))
        mout = zeros(T, dim(Hout), dim(Hout))
        nz = findall(!iszero, ψ)
        splits = Dict(j => split_state(basisstate(j, H), mapper_in) for j in nz)
        _foreach_partial_transpose_term(((j, k) for j in nz for k in nz), H, Hsub, Hout, splits, mapper_out; phase_factors, skipmissing) do J1, J2, K1, K2, v
            mout[K1, K2] += v * ψ[J1] * conj(ψ[J2])
        end
        return mout
    end
    if length(ψ) == dim(H)^2
        return partial_transpose(reshape(ψ, dim(H), dim(H)), H, Hsub; complement, Hout, phase_factors, skipmissing)
    end
    throw(DimensionMismatch("The vector must have length $(dim(H)) (pure state) or $(dim(H)^2) (vectorized density matrix), got $(length(ψ))"))
end

partial_transpose(λ::UniformScaling, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; Hout=H, kwargs...) = Matrix(λ.λ * I(dim(Hout)))                       # R_A(1) = 1, no map needed

## Sparse superoperator

function partial_transpose_map(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement, Hout::AbstractHilbertSpace,
    mapper_in=state_mapper(H, _pt_parts(Hsub, complement)),
    mapper_out=_partial_transpose_out_mapper(H, Hout, Hsub, complement, mapper_in);
    phase_factors=true, skipmissing=false)
    T = _partial_transpose_eltype(H, Hsub, Hout, mapper_in, mapper_out, phase_factors)
    splits = _split_all_states(H, mapper_in)
    indK = LinearIndices((1:dim(Hout), 1:dim(Hout)))
    indJ = LinearIndices((1:dim(H), 1:dim(H)))
    Is = Int[]
    Js = Int[]
    Vs = T[]
    inds = CartesianIndices((dim(H), dim(H)))
    _foreach_partial_transpose_term(inds, H, Hsub, Hout, splits, mapper_out; phase_factors, skipmissing) do J1, J2, K1, K2, v
        push!(Is, indK[K1, K2])
        push!(Js, indJ[J1, J2])
        push!(Vs, v)
    end
    return sparse(Is, Js, Vs, dim(Hout)^2, dim(H)^2)
end
function partial_transpose_map(H, Hsub; complement=complementary_subsystem(H, Hsub), Hout=H, kwargs...)
    partial_transpose_map(H, Hsub, complement, Hout; kwargs...)
end

"""
    PartialTransposeMap

Callable partial-transpose superoperator created by `partial_transpose(H, Hsub; kwargs...)`.
Apply it as `op(m)`, `op(ψ)` (pure state) or `op(out, m)`; get the sparse matrix with `sparse(op)`.
"""
struct PartialTransposeMap{H,HS,C,HO,K,M}
    H::H
    Hsub::HS
    complement::C
    Hout::HO
    kwargs::K
    map::M
end

function PartialTransposeMap(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), Hout=H, kwargs...)
    smap = partial_transpose_map(H, Hsub, complement, Hout; kwargs...)
    PartialTransposeMap(H, Hsub, complement, Hout, kwargs, smap)
end

# |ψ⟩ ↦ map * vec(ψψ†) without forming ψψ† (same as _apply_ptmap_on_vec!, but on a bare matrix)
function _apply_superop_on_pure_state!(w::AbstractVector, mat::SparseMatrixCSC, ψ::AbstractVector)
    D = length(ψ)
    fill!(w, zero(eltype(w)))
    rows = rowvals(mat)
    vals = nonzeros(mat)
    @inbounds for c in 1:size(mat, 2)
        r = nzrange(mat, c)
        isempty(r) && continue
        j = (c - 1) % D + 1
        k = div(c - 1, D) + 1
        val_ψ = ψ[j] * conj(ψ[k])
        iszero(val_ψ) && continue
        for ptr in r
            w[rows[ptr]] += vals[ptr] * val_ψ
        end
    end
    return w
end

function (op::PartialTransposeMap)(in::Union{AbstractMatrix,UniformScaling})
    v = _canonicalize_pt_input(in, op.H)
    reshape(op.map * v, dim(op.Hout), dim(op.Hout))
end
function (op::PartialTransposeMap)(in::AbstractVector)
    if length(in) == dim(op.H)
        w = zeros(promote_type(eltype(op.map), eltype(in)), dim(op.Hout)^2)
        _apply_superop_on_pure_state!(w, op.map, in)
        return reshape(w, dim(op.Hout), dim(op.Hout))
    end
    v = _canonicalize_pt_input(in, op.H)
    reshape(op.map * v, dim(op.Hout), dim(op.Hout))
end
function (op::PartialTransposeMap)(out, in::Union{AbstractMatrix,UniformScaling})
    vout = _canonicalize_pt_output(out, op.Hout)
    vin = _canonicalize_pt_input(in, op.H)
    mul!(vout, op.map, vin)
    out
end
function (op::PartialTransposeMap)(out, in::AbstractVector)
    vout = _canonicalize_pt_output(out, op.Hout)
    if length(in) == dim(op.H)
        _apply_superop_on_pure_state!(vout, op.map, in)
        return out
    end
    vin = _canonicalize_pt_input(in, op.H)
    mul!(vout, op.map, vin)
    out
end
SparseArrays.sparse(op::PartialTransposeMap) = op.map

"""
    logarithmic_negativity(ρ, H, Hsub; kwargs...)

`log ‖ρ^{R_A}‖₁` with the (fermionic) partial transpose on `Hsub`. Uses the sum of singular
values, since `ρ^{R_A}` is not Hermitian for fermions.
"""
logarithmic_negativity(ρ, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; kwargs...) =
    log(sum(svdvals(Matrix(partial_transpose(ρ, H, Hsub; kwargs...)))))

## Tests

@testitem "Fermionic partial transpose: two modes" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: _bit, logarithmic_negativity
    @fermions f
    H = hilbert_space(f, 1:2)
    idx(str) = state_index(str, H)
    ρ = zeros(ComplexF64, 4, 4)
    ρ[idx("10"), idx("01")] = 1 # |10⟩⟨01| = c₁†c₂
    R1 = partial_transpose(ρ, H, hilbert_space(f, [1])) # i c₁c₂ = -i|00⟩⟨11|
    R2 = partial_transpose(ρ, H, hilbert_space(f, [2])) # i c₁†c₂† = i|11⟩⟨00|
    @test R1[idx("00"), idx("11")] ≈ -im && count(!iszero, R1) == 1
    @test R2[idx("11"), idx("00")] ≈ im && count(!iszero, R2) == 1
    T1 = partial_transpose(ρ, H, hilbert_space(f, [1]); phase_factors=false) # ordinary partial transpose
    @test T1[idx("00"), idx("11")] ≈ 1 && count(!iszero, T1) == 1

    ψ = zeros(ComplexF64, 4)
    ψ[idx("10")] = ψ[idx("01")] = 1 / sqrt(2)
    @test partial_transpose(ψ, H, hilbert_space(f, [1])) ≈ partial_transpose(ψ * ψ', H, hilbert_space(f, [1]))
    @test logarithmic_negativity(ψ, H, hilbert_space(f, [1])) ≈ log(2)

    H1 = hilbert_space(f, 1:1, NumberConservation(0))
    ψ1 = ComplexF64[1 + im]
    @test partial_transpose(ψ1, H1, H1) ≈ reshape([abs2(ψ1[1])], 1, 1)
    @test partial_transpose(H1, H1)(ψ1) ≈ reshape([abs2(ψ1[1])], 1, 1)
    @test logarithmic_negativity(ψ, H, hilbert_space(f, [1])) ≈ log(2)
end

@testitem "Fermionic partial transpose: disconnected regions" begin
    using LinearAlgebra, SparseArrays
    using FermionicHilbertSpaces: _bit
    @fermions f
    N = 4
    H = hilbert_space(f, 1:N)
    HA = hilbert_space(f, [1, 3])
    HB = hilbert_space(f, [2, 4])
    nA = zeros(Int, dim(H))
    par = zeros(Bool, dim(H))
    for s in basisstates(H)
        i = state_index(s, H)
        par[i] = iseven(count(j -> _bit(s, j), 1:N))
        nA[i] = count(j -> _bit(s, j), (1, 3))
    end
    pA = Diagonal((-1) .^ nA)
    X = rand(ComplexF64, dim(H), dim(H))
    ρ = X * X'
    ρ = ρ .* (par .== par') # physical (parity-even) state
    ρ ./= tr(ρ)

    RA = partial_transpose(H, HA)
    RB = partial_transpose(H, HB)
    @test RA(ρ) ≈ partial_transpose(ρ, H, HA)          # map == direct
    @test RA(RB(ρ)) ≈ transpose(ρ)                      # R_A ∘ R_B = full transpose (even ops)
    @test RA(RA(ρ)) ≈ pA * ρ * pA                       # R_A² = p_A ⋅ p_A
    @test RA(ρ)' ≈ pA * RA(ρ) * pA                      # pseudo-Hermiticity
    @test sum(svdvals(RA(ρ))) ≈ sum(svdvals(RB(ρ)))     # ‖ρ^{R_A}‖₁ = ‖ρ^{R_B}‖₁

    m = rand(ComplexF64, dim(H), dim(H))
    R13 = partial_transpose(H, hilbert_space(f, [1, 3]))
    R2 = partial_transpose(H, hilbert_space(f, [2]))
    R123 = partial_transpose(H, hilbert_space(f, [1, 2, 3]))
    @test sparse(R13) * sparse(R2) ≈ sparse(R123)       # additive over disjoint pieces
end

@testitem "Fermionic partial transpose: constrained spaces" begin
    @fermions f
    H = hilbert_space(f, 1:4, NumberConservation(2))
    Hfull = hilbert_space(f, 1:4)
    HA = subregion(hilbert_space(f, [1, 3]), H)
    m = rand(ComplexF64, dim(H), dim(H))
    @test_throws ArgumentError partial_transpose(m, H, HA) # leaves the N = 2 sector
    E = zeros(dim(Hfull), dim(H))
    for s in basisstates(H)
        E[state_index(s, Hfull), state_index(s, H)] = 1
    end
    @test partial_transpose(m, H => Hfull, HA) ≈ partial_transpose(E * m * E', Hfull, hilbert_space(f, [1, 3]))
    @test partial_transpose(H, HA; Hout=Hfull)(m) ≈ partial_transpose(m, H, HA; Hout=Hfull)
end

@testitem "Partial transpose: bosons, spins, and Majoranas" begin
    using LinearAlgebra
    @boson b
    @spin s
    @majoranas γ

    Hb = hilbert_space(b, 3)
    Hs = hilbert_space(s, 1 // 2)
    mb = rand(ComplexF64, dim(Hb), dim(Hb))
    ms = rand(ComplexF64, dim(Hs), dim(Hs))

    @test partial_transpose(mb, Hb, Hb) ≈ transpose(mb)
    @test partial_transpose(ms, Hs, Hs) ≈ transpose(ms)
    @test partial_transpose(mb, Hb, Hb; phase_factors=false) ≈ transpose(mb)

    Hγ = hilbert_space(γ, 1:4)
    HγA = subregion(hilbert_space(γ, 1:2), Hγ)
    idx(str) = state_index(str, Hγ)
    mγ = zeros(ComplexF64, dim(Hγ), dim(Hγ))
    mγ[idx("10"), idx("01")] = 1
    Rγ = partial_transpose(mγ, Hγ, HγA)
    @test Rγ[idx("00"), idx("11")] ≈ -im && count(!iszero, Rγ) == 1
end

@testitem "Partial transpose: mixed spaces" begin
    using LinearAlgebra
    @fermions f
    @spin s
    @boson b

    Hf = hilbert_space(f, 1:1)
    Hs = hilbert_space(s, 1 // 2)
    Hb = hilbert_space(b, 3)
    H = tensor_product(Hf, Hs, Hb)
    Hsb = tensor_product(Hs, Hb)
    mf = rand(ComplexF64, dim(Hf), dim(Hf))
    ms = rand(ComplexF64, dim(Hs), dim(Hs))
    mb = rand(ComplexF64, dim(Hb), dim(Hb))
    m = kron(reverse([mf, ms, mb])...)

    expected = kron(transpose(mb), transpose(ms), mf)
    @test partial_transpose(m, H, Hsb) ≈ expected
end
