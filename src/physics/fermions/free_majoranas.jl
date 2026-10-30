struct MajoranaSingleParticleHilbertSpace{D}
    majoranaindices::D
end
dim(H::MajoranaSingleParticleHilbertSpace) = length(H.majoranaindices)
mode_ordering(H::MajoranaSingleParticleHilbertSpace) = H.majoranaindices
modes(H::MajoranaSingleParticleHilbertSpace) = keys(H.majoranaindices)
Base.:(==)(a::MajoranaSingleParticleHilbertSpace, b::MajoranaSingleParticleHilbertSpace) = a.majoranaindices == b.majoranaindices
Base.hash(x::MajoranaSingleParticleHilbertSpace, h::UInt) = hash(x.majoranaindices, h)
Base.show(io::IO, H::MajoranaSingleParticleHilbertSpace) = print(io, "MajoranaSingleParticleHilbertSpace($(dim(H)) Majoranas)")

"""
    hilbert_space(γ::SymbolicMajoranaBasis, labels, ::SingleParticle)

The space of the Majoranas `γ[l]`, `l in labels`, for quadratic Majorana operators. `representation(op, H)` is the antisymmetric matrix `R` with `op = Σᵢⱼ R[i, j] γ[i] γ[j] / 2` up to a constant.

```julia
@majoranas γ
H = hilbert_space(γ, 1:2, SingleParticle())
representation(1im * γ[1] * γ[2], H, :dense) # [0 1im; -1im 0]
```
"""
function hilbert_space(γ::SymbolicMajoranaBasis, labels::AbstractVector, ::SingleParticle)
    iseven(length(labels)) || throw(ArgumentError("The number of Majoranas must be even, got $(length(labels))."))
    return MajoranaSingleParticleHilbertSpace(OrderedDict(γ[l] => n for (n, l) in enumerate(labels)))
end

representation(op, H::MajoranaSingleParticleHilbertSpace, repr = EagerSparseRepr(); chunking = NoChunking(), kwargs...) = matrix_representation(op, H, repr; chunking = chunking, kwargs...)
function matrix_representation(op, H::MajoranaSingleParticleHilbertSpace, repr = EagerSparseRepr(); chunking = NoChunking(), kwargs...)
    _isscalar_operator(op) && return _identity_free_zero(H, repr)
    _is_quadratic_operator(op, (a, b) -> a isa MajoranaSym && b isa MajoranaSym) || throw(ArgumentError("Only quadratic Majorana operators can be represented on a MajoranaSingleParticleHilbertSpace. Use `hilbert_space(γ, labels)` and `representation(op, H)` for other operators."))
    m = _quadratic_matrix(remove_identity(op), H, repr) do term, H
        a, b = term.factors
        return mode_position(a, H), mode_position(b, H), term.coeff
    end
    return m - permutedims(m)
end

@testitem "Majorana single-particle space" begin
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra
    @majoranas γ
    N = 6
    H = hilbert_space(γ, 1:N, SingleParticle())
    Hmb = hilbert_space(γ, 1:N)
    op = sum(randn(ComplexF64) * γ[i] * γ[j] for i in 1:N, j in 1:N)
    R = representation(op, H, :dense)
    @test R == representation(op, H)
    rep(x) = representation(x, Hmb, :dense)
    for k in 1:N
        @test rep(op * γ[k] - γ[k] * op) ≈ -2 * sum(R[k, j] * rep(γ[j]) for j in 1:N)
    end
    @test representation(symbolic_operator(R, H), H) ≈ R
    @test iszero(representation(2 + 0 * γ[1], H))
    @test_throws ArgumentError representation(γ[1] * γ[2] * γ[3] * γ[4], H)
    @test_throws ArgumentError representation(γ[1] * γ[7], H)
    @test_throws ArgumentError hilbert_space(γ, 1:3, SingleParticle())
end
