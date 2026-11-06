const FreeFermionSpace = Union{SingleParticleHilbertSpace, BdGHilbertSpace, MajoranaSingleParticleHilbertSpace}

# The scale `s` in `[op, Φ_k] = (1/s) Σ_j rep(op)[j, k] Φ_j`, with Φ = c' (single particle), Ψ' (BdG) and γ (Majorana)
_commutator_scale(::SingleParticleHilbertSpace) = 1
_commutator_scale(::Union{BdGHilbertSpace, MajoranaSingleParticleHilbertSpace}) = 1 // 2

struct FreeProductSpace{F <: FreeFermionSpace, A <: AbstractHilbertSpace}
    free::F
    aux::A
end

"""
    tensor_product(Hfree, Haux)

The quasiparticles of the free-fermion space `Hfree` (`SingleParticle()`, `BdG()` or Majorana) coupled to
another system with space `Haux`. Index `i` of `Hfree` and `a` of `Haux` is index `(a - 1) * dim(Hfree) + i`,
the order of `kron(X, Q)`.

`representation(op, H)` takes `op = Σₖ Qₖ Xₖ + Σⱼ Yⱼ`, with `Qₖ` quadratic in the fermions and `Xₖ`, `Yⱼ` acting on
`Haux`. It gives `Σₖ representation(Qₖ, Hfree) ⊗ Xₖ` plus `s Yⱼ` for each `Yⱼ` that does not commute with all
couplings `Xₖ`, where `s` is the scale of the `Hfree` matrix (`1/2` for `BdG()` and Majoranas, `1` for
`SingleParticle()`). A `Yⱼ` that commutes with all couplings, such as a constant, does not change the quasiparticles
and is dropped. A kept `Yⱼ` should be zero on the state of `Haux` that the quasiparticles are expanded around;
adding a constant to it shifts all eigenvalues.

`representation` throws an `ArgumentError` unless the fermions stay free: the couplings must commute with each other,
and each commutator of a kept `Yⱼ` with a coupling must commute with all couplings.
"""
tensor_product(Hfree::FreeFermionSpace, Haux::AbstractHilbertSpace) = FreeProductSpace(Hfree, Haux)
dim(H::FreeProductSpace) = dim(H.free) * dim(H.aux)
Base.:(==)(a::FreeProductSpace, b::FreeProductSpace) = a.free == b.free && a.aux == b.aux
Base.hash(H::FreeProductSpace, h::UInt) = hash(H.aux, hash(H.free, h))
function Base.show(io::IO, H::FreeProductSpace)
    print(io, "$(dim(H))-dimensional FreeProductSpace(")
    show(IOContext(io, :compact => true), H.free)
    print(io, " ⊗ ")
    show(IOContext(io, :compact => true), H.aux)
    return print(io, ")")
end

_isfree_factor(x) = x isa Union{FermionSym, MajoranaSym}
_commutator(a, b) = a * b - b * a
_iszero_operator(x::Number) = iszero(x)
_iszero_operator(x::UniformScaling) = iszero(x.λ)
_iszero_operator(x::Union{NCMul, NCAdd}) = all(t -> iszero(t.coeff), NCterms(x)) && (!(x isa NCAdd) || iszero(x.coeff))
_iszero_operator(x) = false
_commutes(a, b) = _iszero_operator(_commutator(a, b))

function representation(op, H::FreeProductSpace, repr = EagerSparseRepr(); kwargs...)
    return matrix_representation(op, H, repr; kwargs...)
end
function matrix_representation(op, H::FreeProductSpace, repr = EagerSparseRepr(); kwargs...)
    repr isa Symbol && return matrix_representation(op, H, _canonicalize_repr(repr); kwargs...)
    _repr = _eager_repr(repr, H)
    nf, na = dim(H.free), dim(H.aux)
    S = spzeros(Int, dim(H), dim(H))
    _isscalar_operator(op) && return _finalize_free_product(S, _repr)
    couplings = Any[]
    quadratic = Tuple{Any, Any}[]
    others = Any[]
    for term in NCterms(remove_identity(op))
        Q = filter(_isfree_factor, term.factors)
        X = filter(!_isfree_factor, term.factors)
        Xop = isempty(X) ? 1 : prod(X)
        if isempty(Q)
            push!(others, term.coeff * Xop)
        elseif length(Q) == 2
            push!(quadratic, (term.coeff * prod(Q), Xop))
            any(isequal(Xop), couplings) || push!(couplings, Xop)
        else
            throw(ArgumentError("Only terms quadratic in the fermions can be represented on a FreeProductSpace, got $term."))
        end
    end
    for (i, X) in enumerate(couplings), X2 in couplings[(i + 1):end]
        _commutes(X, X2) || throw(ArgumentError("The couplings $X and $X2 do not commute, so the fermions are not free."))
    end
    aux_matrix(X) = X isa Number ? X * sparse(I, na, na) : representation(X, H.aux, EagerSparseRepr(); projection = true)
    for (Q, X) in quadratic
        S += kron(aux_matrix(X), representation(Q, H.free, EagerSparseRepr()))
    end
    for Y in others
        all(X -> _commutes(Y, X), couplings) && continue
        for X in couplings, X2 in couplings
            _commutes(_commutator(Y, X), X2) ||
                throw(ArgumentError("The commutator of $Y with the coupling $X does not commute with the coupling $X2, so the fermions are not free."))
        end
        S += _commutator_scale(H.free) * kron(aux_matrix(Y), sparse(I, nf, nf))
    end
    return _finalize_free_product(S, _repr)
end
_finalize_free_product(S, ::EagerSparseRepr) = S
_finalize_free_product(S, ::EagerDenseRepr) = Matrix(S)

@testitem "Free fermions coupled to another system" begin
    using FermionicHilbertSpaces.FreeFermions, LinearAlgebra
    import FermionicHilbertSpaces: AbstractSym, AbstractBasisState, local_mul_effect, apply_local_operator, symbolic_group
    import FermionicHilbertSpaces.NonCommutativeProducts: @nc
    # A shift operator T|n⟩ = |n + 1⟩ and a counting operator K|n⟩ = n|n⟩, with [K, Tᵐ] = m Tᵐ
    struct TBasis end
    struct TShift <: AbstractSym
        shift::Int
    end
    struct TCount <: AbstractSym
        power::Int
    end
    Base.adjoint(T::TShift) = TShift(-T.shift)
    Base.adjoint(K::TCount) = K
    @nc TShift TCount
    local_mul_effect(a::TShift, b::TShift) = a.shift + b.shift == 0 ? 1 : TShift(a.shift + b.shift)
    local_mul_effect(a::TCount, b::TCount) = TCount(a.power + b.power)
    local_mul_effect(a::TShift, b::TCount) = b.power == 1 ? b * a - a.shift * a : TCount(1) * a * TCount(b.power - 1) - a.shift * a * TCount(b.power - 1)
    struct TState <: AbstractBasisState
        n::Int
    end
    apply_local_operator(op::TShift, s::TState, space, precomp) = TState(s.n + op.shift), 1
    apply_local_operator(op::TCount, s::TState, space, precomp) = s, s.n^op.power
    symbolic_group(::Union{TShift, TCount}) = TBasis()
    symbolic_group(b::TBasis) = b

    nmax, ω = 6, 2.3
    HA = GenericHilbertSpace(TBasis(), TState.(-nmax:nmax))
    T, K = TShift(1), TCount(1)
    fold(ε) = mod(ε + ω / 2, ω) - ω / 2
    # eigenvalues closest to zero, one per state of the free space
    central(vals, n) = vals[sort(partialsortperm(abs.(vals), 1:n))]
    # folded differences of many-body eigenvalues between the parity sectors
    function manybody_differences(Hmb, op)
        H = tensor_product(Hmb, HA; constraint = ParityConservation(Hmb))
        εs = map(sectors(H)) do S
            central(eigvals(Hermitian(Matrix(representation(op, S, :dense; projection = true)))), dim(S) ÷ dim(HA))
        end
        return vec([fold(a - b) for a in εs[1], b in εs[2]])
    end
    contained(a, b) = all(x -> minimum(abs.(b .- x)) < 1.0e-6, a)
    dropped = 0.37 + 0.21 * (T + T') # commutes with all couplings
    P = reverse(Matrix(I, 2nmax + 1, 2nmax + 1), dims = 1) # n → -n

    @fermions c
    op = ω * K + 0.3 * c[1]'c[1] - 0.2 * c[2]'c[2] + (0.9 * c[1]'c[2] + 0.6 * c[1]'c[2]' + hc) +
        (0.4 * c[1]'c[1] * T + 0.15im * c[1]'c[2]' * T + 0.1 * c[2] * c[1] * T^2 + hc) + dropped
    Hbdg = hilbert_space(c, 1:2, BdG())
    H = tensor_product(Hbdg, HA)
    @test dim(H) == 4 * (2nmax + 1)
    S = representation(op, H, :dense)
    @test S ≈ representation(op, H)
    @test S ≈ representation(op - dropped, H, :dense)
    @test contained(central(2 * eigvals(Hermitian(S)), 4), manybody_differences(hilbert_space(c, 1:2), op))
    C = kron(P, [0I I(2); I(2) 0I])
    @test -C * transpose(S) * C ≈ S

    opsp = ω * K + 0.3 * c[1]'c[1] + (0.9 * c[1]'c[2] + hc) + (0.5 * c[1]'c[1] * T + 0.2im * c[1]'c[2] * T^2 + hc) + dropped
    Ssp = representation(opsp, tensor_product(hilbert_space(c, 1:2, SingleParticle()), HA), :dense)
    @test contained(central(eigvals(Hermitian(Ssp)), 2), manybody_differences(hilbert_space(c, 1:2), opsp))

    @majoranas γ
    opm = ω * K + 0.5im * γ[1] * γ[2] + 0.4im * γ[2] * γ[3] + 0.3im * γ[3] * γ[4] +
        (0.35im * γ[1] * γ[3] * T + 0.2im * γ[2] * γ[3] * T + hc) + dropped
    Sm = representation(opm, tensor_product(hilbert_space(γ, 1:4, SingleParticle()), HA), :dense)
    @test contained(central(2 * eigvals(Hermitian(Sm)), 4), manybody_differences(hilbert_space(γ, 1:4), opm))
    @test -kron(P, I(4)) * conj(Sm) * kron(P, I(4)) ≈ Sm

    # A conserved coupling: one free problem per eigenvalue of σ[1][:z]; terms on the spins alone drop out
    @spins σ 1 // 2
    Hspins = hilbert_space(σ, 1:2)
    sector(λ) = (0.9 * λ * c[1]'c[2] + 0.6 * c[1]'c[2]' + hc) + 0.3 * λ * c[1]'c[1] - 0.25 * c[2]'c[2]
    opz = (0.9 * σ[1][:z] * c[1]'c[2] + 0.6 * c[1]'c[2]' + hc) + 0.3 * σ[1][:z] * c[1]'c[1] - 0.25 * c[2]'c[2] +
        0.4 * σ[1][:z] + 0.7 * σ[2][:x] + 0.37
    Sz = representation(opz, tensor_product(Hbdg, Hspins), :dense)
    @test sort(eigvals(Hermitian(Sz))) ≈ sort(vcat([eigvals(Hermitian(representation(sector(λ), Hbdg, :dense))) for λ in (-1, -1, 1, 1) ./ 2]...))

    @test iszero(representation(dropped, H))
    @test_throws ArgumentError representation(op + K^2, H)                      # [K², T] does not commute with T
    @test_throws ArgumentError representation(op + c[1]'c[1] * K, H)            # couplings K and T do not commute
    @test_throws ArgumentError representation(c[1]'c[1] * c[2]'c[2], H)        # not quadratic
    @test_throws ArgumentError representation(opz + 0.2 * σ[1][:x], tensor_product(Hbdg, Hspins))
end
