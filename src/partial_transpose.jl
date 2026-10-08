
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

partial_transpose_phase_factor_eltype(H::AbstractHilbertSpace) = partial_trace_phase_factor_eltype(H) # default assumption 

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
        s_in = phase_factors ? conj(partial_trace_phase_factor(f1, f2, H)) : 1 # Fock(H) → operator basis
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
pass a larger output space on the same modes as `Hout`. For fermions the result is not
Hermitian: `(ρ^{R_A})' == p_A ρ^{R_A} p_A` for parity-even `ρ`, with `p_A` the parity of `Hsub`.
"""
function partial_transpose(m::AbstractMatrix, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace;
    complement=complementary_subsystem(H, Hsub), Hout=H, phase_factors=true, skipmissing=false)
    mapper_in = state_mapper(H, _pt_parts(Hsub, complement))
    mapper_out = _partial_transpose_out_mapper(H, Hout, Hsub, complement, mapper_in)
    T = promote_type(eltype(m), partial_transpose_phase_factor_eltype(H))
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
        T = promote_type(eltype(ψ), partial_transpose_phase_factor_eltype(H))
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

partial_transpose(λ::UniformScaling, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; Hout=H, kwargs...) =
    partial_transpose(λ.λ * I(dim(H)), H, Hsub; Hout, kwargs...)

## Sparse superoperator

function partial_transpose_map(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement, Hout::AbstractHilbertSpace,
    mapper_in=state_mapper(H, _pt_parts(Hsub, complement)),
    mapper_out=_partial_transpose_out_mapper(H, Hout, Hsub, complement, mapper_in);
    phase_factors=true, skipmissing=false)
    T = partial_transpose_phase_factor_eltype(H)
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
    return SparseArrays.sparse!(Is, Js, Vs, dim(Hout)^2, dim(H)^2)
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
    _devectorize_pt_output(op.map * v, dim(op.Hout))
end
function (op::PartialTransposeMap)(in::AbstractVector)
    if length(in) == dim(op.H)
        w = zeros(promote_type(eltype(op.map), eltype(in)), dim(op.Hout)^2)
        _apply_superop_on_pure_state!(w, op.map, in)
        return reshape(w, dim(op.Hout), dim(op.Hout))
    end
    v = _canonicalize_pt_input(in, op.H)
    _devectorize_pt_output(op.map * v, dim(op.Hout))
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
    using FermionicHilbertSpaces: logarithmic_negativity
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
    ψ1 = ComplexF64[1+im]
    @test partial_transpose(ψ1, H1, H1) ≈ reshape([abs2(ψ1[1])], 1, 1)
    @test partial_transpose(H1, H1)(ψ1) ≈ reshape([abs2(ψ1[1])], 1, 1)
end

@testitem "Fermionic partial transpose: constrained spaces" begin
    using LinearAlgebra
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
    @test partial_transpose(2I, H => Hfull, HA) ≈ partial_transpose(2I(dim(H)), H => Hfull, HA)
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

@testmodule PartialTranspose begin
    using LinearAlgebra, SparseArrays, Random, Test
    using FermionicHilbertSpaces
    using FermionicHilbertSpaces: nbr_of_modes
    export PTFamily, check_partial_transpose, check_tensor_compatibility,
        tracenorm, rand_density_matrix, charge_project

    """
        PTFamily(; name, basis, rand_state, kwargs=(;), gauge=nothing, skip=())

    Partial-transpose family on the modes of the symbolic `basis`.
    * `rand_state(rng, H)`: random physical density matrix (parity-even / charge-neutral).
    * `kwargs`: passed to `partial_transpose` and `tensor_product`.
    * `gauge(H, HA)`: `G` with `R_A² = Ad_G` and `R_A(ρ)' = G R_A(ρ) G'` (`I`: Hermitian), or `nothing`.
    * `skip`: properties the family violates; pin each with an explicit counterexample test.
      Names: `:isometry, :local, :trace, :composition, :full_transpose, :symmetry, :unitary, :separable`.
    """
    Base.@kwdef struct PTFamily{B,R,K<:NamedTuple,G}
        name::String
        basis::B
        rand_state::R
        kwargs::K = (;)
        gauge::G = nothing
        skip::Tuple{Vararg{Symbol}} = ()
    end

    tracenorm(X) = sum(svdvals(Matrix(X)))
    rand_op(rng, H) = randn(rng, ComplexF64, dim(H), dim(H))
    rand_density_matrix(rng, H) = (X = rand_op(rng, H); ρ = X * X'; ρ / tr(ρ))
    "Keep the blocks of `ρ` that commute with the diagonal charge operator `Q`."
    charge_project(ρ, Q) = (q = Vector(diag(Q)); ρ .* isapprox.(q, permutedims(q)))
    rand_unitary(fam, rng, H) = exp(10im * Matrix(fam.rand_state(rng, H)))   # physical
    bell(fam, H, a, b) = Vector(representation((fam.basis[a]' + fam.basis[b]') * Ket("0", H))) / √2
    eye(H) = Matrix{ComplexF64}(I, dim(H), dim(H))

    """
        check_partial_transpose(fam, H, A; B=complement, rng=Xoshiro(1), atol=1e-8)

    Properties P1–P12 for the cut `A | B` of the modes of `H`.
    """
    function check_partial_transpose(fam::PTFamily, H, A;
        B=setdiff(1:nbr_of_modes(H), A), rng=Xoshiro(1), atol=1e-8)
        (isempty(A) || isempty(B)) && throw(ArgumentError("A and B must both be non-empty"))
        kw = fam.kwargs
        HA, HB = hilbert_space(fam.basis, A), hilbert_space(fam.basis, B)
        RA, RB, RAB = (partial_transpose(H, S; kw...) for S in (HA, HB, H))
        ⊗(x, y) = tensor_product((x, y), (HA, HB), H; kw...)
        X, Y = rand_op(rng, H), rand_op(rng, H)
        ρ = fam.rand_state(rng, H)
        ψ = bell(fam, H, first(A), first(B))
        G = isnothing(fam.gauge) ? nothing : fam.gauge(H, HA)
        check(p) = p ∉ fam.skip

        @testset "$(fam.name): A = $A" begin
            # P1 linearity / implementation consistency: sparse map == direct kernel, pure-state path
            @test RA(X) ≈ partial_transpose(X, H, HA; kw...) atol = atol
            @test RA(ψ) ≈ RA(ψ * ψ') atol = atol
            # P2 Hilbert–Schmidt isometry
            check(:isometry) && @test dot(RA(X), RA(Y)) ≈ dot(X, Y) atol = atol
            # P3 trivial on the complement (4.2)
            if check(:local)
                for XB in (eye(HB), fam.rand_state(rng, HB))
                    @test RA(eye(HA) ⊗ XB) ≈ eye(HA) ⊗ XB atol = atol
                end
            end
            # P4 trace preservation
            check(:trace) && @test tr(RA(X)) ≈ tr(X) atol = atol
            # P5, P6 gauge: R_A² = Ad_G and R_A(ρ)' = G R_A(ρ) G'
            if !isnothing(G)
                @test RA(RA(X)) ≈ G * X * G' atol = atol
                @test RA(ρ)' ≈ G * RA(ρ) * G' atol = atol
            end
            # P7 composition over disjoint regions
            if check(:composition)
                @test sparse(RA) * sparse(RB) ≈ sparse(RAB) atol = atol
                if length(A) ≥ 2
                    R1, R2 = (partial_transpose(H, hilbert_space(fam.basis, S); kw...) for S in (A[1:1], A[2:end]))
                    @test sparse(R1) * sparse(R2) ≈ sparse(RA) atol = atol
                end
            end
            # P8 the full transpose is the transpose on physical states (D2)
            if check(:full_transpose)
                @test RAB(ρ) ≈ transpose(ρ) atol = atol
                @test svdvals(RAB(ρ)) ≈ svdvals(ρ) atol = atol
            end
            # P9 ‖R_A ρ‖₁ = ‖R_B ρ‖₁ (D1)
            check(:symmetry) && @test tracenorm(RA(ρ)) ≈ tracenorm(RB(ρ)) atol = atol
            # P10 local-unitary invariance (4.10)
            if check(:unitary)
                U = rand_unitary(fam, rng, HA) ⊗ rand_unitary(fam, rng, HB)
                @test tracenorm(RA(U * ρ * U')) ≈ tracenorm(RA(ρ)) atol = atol
            end
            # P11 separable ⇒ ‖R_A σ‖₁ = 1 (4.7); positive semidefinite if Hermitian
            if check(:separable)
                σ = sum(fam.rand_state(rng, HA) ⊗ fam.rand_state(rng, HB) for _ in 1:3) / 3
                @test tracenorm(RA(σ)) ≈ 1 atol = atol
                G isa UniformScaling && @test eigmin(Hermitian(Matrix(RA(σ)))) ≥ -atol
            end
            # P12 a Bell pair across the cut gives ‖R_A ρ‖₁ = 2
            @test tracenorm(RA(ψ)) ≈ 2 atol = atol
        end
    end

    """
        check_tensor_compatibility(fam, (H1, A1), (H2, A2), H; rng=Xoshiro(1), atol=1e-8)

    P13 (4.15): `R_{A1∪A2}(ρ1 ⊗ ρ2) = R_{A1}(ρ1) ⊗ R_{A2}(ρ2)` for `H = H1 ⊗ H2`.
    """
    function check_tensor_compatibility(fam::PTFamily, (H1, A1), (H2, A2), H; rng=Xoshiro(1), atol=1e-8)
        kw = fam.kwargs
        ρ1, ρ2 = fam.rand_state(rng, H1), fam.rand_state(rng, H2)
        R1 = partial_transpose(H1, hilbert_space(fam.basis, A1); kw...)
        R2 = partial_transpose(H2, hilbert_space(fam.basis, A2); kw...)
        R = partial_transpose(H, hilbert_space(fam.basis, [A1; A2]); kw...)
        ⊗(x, y) = tensor_product((x, y), (H1, H2), H; kw...)
        @testset "$(fam.name): ⊗-compatibility, A = $A1 ∪ $A2" begin
            @test R(ρ1 ⊗ ρ2) ≈ R1(ρ1) ⊗ R2(ρ2) atol = atol
        end
    end
end

@testitem "Partial transpose properties: standard and fermionic" setup = [PartialTranspose] begin
    using FermionicHilbertSpaces: logarithmic_negativity
    using LinearAlgebra
    @fermions f
    H = hilbert_space(f, 1:4)
    H12, H34 = hilbert_space(f, 1:2), hilbert_space(f, 3:4)

    rand_even_state(rng, H) = charge_project(rand_density_matrix(rng, H), parityoperator(H))
    parity_gauge(H, HA) = embed(Matrix(parityoperator(HA)), HA, H)
    families = (
        PTFamily(; name="fermionic", basis=f, rand_state=rand_even_state, gauge=parity_gauge),
        PTFamily(; name="standard", basis=f, rand_state=rand_density_matrix,
            kwargs=(; phase_factors=false), gauge=(H, HA) -> I))

    for fam in families
        for A in ([1], [2], [1, 2], [1, 3], [2, 4], [1, 2, 3])   # single, contiguous, disconnected
            check_partial_transpose(fam, H, A)
        end
        check_tensor_compatibility(fam, (H12, [1]), (H34, [3]), H)
        check_tensor_compatibility(fam, (H12, [1, 2]), (H34, [4]), H)  # full transpose on a factor (4.2)
    end

    @testset "closed-form fermionic negativities" begin
        H2, H1 = hilbert_space(f, 1:2), hilbert_space(f, 1:1)
        pure(ψ) = (v = representation(ψ); v * v')
        bell = pure((Ket("10", H2) + Ket("01", H2)) / √2)
        @test logarithmic_negativity(bell, H2, H1) ≈ log(2)
        @test logarithmic_negativity(bell, H2, hilbert_space(f, 2:2)) ≈ log(2)
        for θ in (0.1, π / 4, 1.0)   # cosθ|00⟩ + sinθ|11⟩
            ρ = pure(cos(θ) * Ket("00", H2) + sin(θ) * Ket("11", H2))
            @test logarithmic_negativity(ρ, H2, H1) ≈ 2log(cos(θ) + sin(θ))
        end
        M = representation(f[1]' * f[2] + f[2]' * f[1], H2, :dense)
        for β in (0.3, 1.0, 2.5)     # thermal hopping state
            ρ = exp(-β * M)
            ρ /= tr(ρ)
            @test logarithmic_negativity(ρ, H2, H1) ≈ log(2cosh(β) / (1 + cosh(β)))
        end
    end
end
