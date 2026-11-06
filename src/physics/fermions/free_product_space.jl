"""
    state_product(a, b)

Extension hook for [`tensor_product`](@ref) of a free-fermion space with an auxiliary space: the
product of the coefficient-algebra elements labelled by the basis states `a` and `b`, as a basis state.
For a Floquet space with states `|n⟩ ↔ e^{inωt}`, `state_product(|m⟩, |n⟩) = |m + n⟩`.
The product must be commutative, with [`unit_state`](@ref) as its unit.
"""
function state_product end

"""
    unit_state(H)

Extension hook: the basis state of the auxiliary space `H` that is the unit of [`state_product`](@ref),
such as `|0⟩` of a Floquet space.
"""
function unit_state end

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

The space of quasiparticles of the free-fermion space `Hfree` (`SingleParticle()`, `BdG()` or Majorana)
dressed by the auxiliary space `Haux`, such as the photon space of a Floquet problem.

`representation(op, H)` takes `op = Σ_k Q_k X_k + Y`, with `Q_k` quadratic in the fermions and `X_k`, `Y` acting
on `Haux`, and gives `Σ_k representation(Q_k, Hfree) ⊗ X_k` plus the action of `Y` on the coefficients, scaled as
the matrix of `Hfree`. For a Floquet operator `ω N + Σ_m H⁽ᵐ⁾ Fᵐ`, this is the Sambe matrix of the
single-particle problem; on a `BdG()` space its eigenvalues are half the quasienergies, as for a static `BdG()` matrix.
Index `i` of `Hfree` and `a` of `Haux` is index `(a - 1) * dim(Hfree) + i`, the order of `kron(X, Q)`.

The basis states of `Haux` must label a commutative algebra through [`state_product`](@ref) and
[`unit_state`](@ref). An `op` is accepted only when it describes free fermions: every `X_k` acts as a
multiplication in that algebra (like `Fᵐ`), and `Y` acts as a derivation of it (like `N`). Otherwise
`representation` throws an `ArgumentError`. Terms of `Y` that are multiplications, such as constants, drop out.
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
    print(io, ")")
end

_isfree_factor(x) = x isa Union{FermionSym, MajoranaSym}

# Amplitudes of `x * |state⟩` as a dictionary from basis states to amplitudes. Exact: no truncation to the space.
function _apply_to_state(x, state, aux)
    ket = SymbolicState(aux, state, nothing)
    result = x * ket
    amps = Dict{typeof(state), Any}()
    result isa Number && (iszero(result) ? (return amps) : throw(ArgumentError("Unexpected scalar result of applying $x to $state.")))
    for term in NCterms(result)
        length(term.factors) == 1 || throw(ArgumentError("Auxiliary spaces with several groups are not supported."))
        s = only(term.factors).ket
        amps[s] = get(amps, s, 0) + term.coeff
    end
    return filter!(p -> !iszero(last(p)), amps)
end
# Σ amp * (s ⋅ b) and Σ amp * (b ⋅ s)
_times(amps, b) = _accumulate(state_product(s, b) => amp for (s, amp) in amps)
function _accumulate(pairs)
    out = Dict{Any, Any}()
    for (s, amp) in pairs
        out[s] = get(out, s, 0) + amp
    end
    return filter!(p -> !iszero(last(p)), out)
end
_approx_equal(a, b; atol) = all(s -> isapprox(get(a, s, 0), get(b, s, 0); atol), union(keys(a), keys(b)))

function representation(op, H::FreeProductSpace, repr = EagerSparseRepr(); kwargs...)
    return matrix_representation(op, H, repr; kwargs...)
end
function matrix_representation(op, H::FreeProductSpace, repr = EagerSparseRepr(); atol = sqrt(eps()), kwargs...)
    repr isa Symbol && return matrix_representation(op, H, _canonicalize_repr(repr); atol, kwargs...)
    _repr = _eager_repr(repr, H)
    aux = H.aux
    states = collect(basisstates(aux))
    index = Dict(s => i for (i, s) in enumerate(states))
    u = unit_state(aux)
    haskey(index, u) || throw(ArgumentError("unit_state($aux) = $u is not a basis state of the auxiliary space."))
    for a in states
        state_product(u, a) == a || throw(ArgumentError("unit_state is not the unit of state_product: $u ⋅ $a ≠ $a."))
        for b in states
            state_product(a, b) == state_product(b, a) || throw(ArgumentError("state_product is not commutative: $a ⋅ $b ≠ $b ⋅ $a."))
        end
    end
    # Matrix of the action `b ↦ f(b)` on the basis of `aux`, dropping states outside it
    function aux_matrix(f)
        rows, cols, vals = Int[], Int[], Any[]
        for (j, b) in enumerate(states), (s, amp) in f(b)
            i = get(index, s, 0)
            iszero(i) || (push!(rows, i); push!(cols, j); push!(vals, amp))
        end
        return sparse(rows, cols, isempty(vals) ? Int[] : identity.(vals), length(states), length(states))
    end
    nf = dim(H.free)
    S = spzeros(Int, dim(H), dim(H))
    Y = 0
    _isscalar_operator(op) && return _finalize_free_product(S, _repr)
    for term in NCterms(remove_identity(op))
        Q = filter(_isfree_factor, term.factors)
        X = filter(!_isfree_factor, term.factors)
        Xop = isempty(X) ? 1 : prod(X)
        if isempty(Q)
            Y += term.coeff * Xop
        elseif length(Q) == 2
            Xu = _apply_to_state(Xop, u, aux)
            for b in states
                _approx_equal(_apply_to_state(Xop, b, aux), _times(Xu, b); atol) ||
                    throw(ArgumentError("In the term $term, the auxiliary part $Xop is not a multiplication operator of the coefficient algebra, so the fermions are not free."))
            end
            S += kron(aux_matrix(b -> _times(Xu, b)), representation(term.coeff * prod(Q), H.free, EagerSparseRepr()))
        else
            throw(ArgumentError("Only terms quadratic in the fermions can be represented on a FreeProductSpace, got $term."))
        end
    end
    if !iszero(Y)
        Yu = _apply_to_state(Y, u, aux)
        ad(b) = _accumulate(Iterators.flatten((_apply_to_state(Y, b, aux), (s => -amp for (s, amp) in _times(Yu, b)))))
        for a in states, b in states
            leibniz = _accumulate(Iterators.flatten((_times(ad(a), b), _times(ad(b), a))))
            _approx_equal(ad(state_product(a, b)), leibniz; atol) ||
                throw(ArgumentError("The part $Y acting only on the auxiliary space is not a derivation of the coefficient algebra, so the fermions are not free."))
        end
        S += _commutator_scale(H.free) * kron(aux_matrix(ad), sparse(I, nf, nf))
    end
    return _finalize_free_product(S, _repr)
end
_finalize_free_product(S, ::EagerSparseRepr) = S
_finalize_free_product(S, ::EagerDenseRepr) = Matrix(S)

@testitem "Free fermions coupled to a Floquet space" begin
    using FermionicHilbertSpaces.FreeFermions, LinearAlgebra
    import FermionicHilbertSpaces: AbstractSym, AbstractBasisState, local_mul_effect, apply_local_operator, symbolic_group
    import FermionicHilbertSpaces.NonCommutativeProducts: @nc
    import FermionicHilbertSpaces.FreeFermions: state_product, unit_state
    # Floquet algebra, as in examples/floquet_tutorial.jl
    struct TFloquetBasis end
    struct TLadder <: AbstractSym
        shift::Int
    end
    struct TNumber <: AbstractSym
        power::Int
    end
    Base.adjoint(F::TLadder) = TLadder(-F.shift)
    Base.adjoint(N::TNumber) = N
    @nc TLadder TNumber
    local_mul_effect(a::TLadder, b::TLadder) = a.shift + b.shift == 0 ? 1 : TLadder(a.shift + b.shift)
    local_mul_effect(a::TNumber, b::TNumber) = TNumber(a.power + b.power)
    local_mul_effect(a::TLadder, b::TNumber) = b.power == 1 ? b * a - a.shift * a : TNumber(1) * a * TNumber(b.power - 1) - a.shift * a * TNumber(b.power - 1)
    struct TState <: AbstractBasisState
        n::Int
    end
    apply_local_operator(op::TLadder, s::TState, space, precomp) = TState(s.n + op.shift), 1
    apply_local_operator(op::TNumber, s::TState, space, precomp) = s, s.n^op.power
    symbolic_group(::Union{TLadder, TNumber}) = TFloquetBasis()
    symbolic_group(b::TFloquetBasis) = b
    state_product(a::TState, b::TState) = TState(a.n + b.n)
    unit_state(::GenericHilbertSpace{TState}) = TState(0)

    nmax, ω = 6, 2.3
    Hq = GenericHilbertSpace(TFloquetBasis(), TState.(-nmax:nmax))
    F, N = TLadder(1), TNumber(1)
    fold(ε) = mod(ε + ω / 2, ω) - ω / 2
    # quasienergies closest to zero, one per state of the free space
    central(vals, n) = vals[sort(partialsortperm(abs.(vals), 1:n))]
    # folded differences of many-body quasienergies between the parity sectors
    function manybody_differences(Hmb, K)
        H = tensor_product(Hmb, Hq; constraint = ParityConservation(Hmb))
        εs = map(sectors(H)) do S
            central(eigvals(Hermitian(Matrix(representation(K, S, :dense; projection = true)))), dim(S) ÷ dim(Hq))
        end
        return vec([fold(a - b) for a in εs[1], b in εs[2]])
    end
    contained(a, b) = all(x -> minimum(abs.(b .- x)) < 1e-6, a)
    cnumber = 0.37 + 0.21 * (F + F')
    P = reverse(Matrix(I, 2nmax + 1, 2nmax + 1), dims = 1) # n → -n

    @fermions c
    K = ω * N + 0.3 * c[1]'c[1] - 0.2 * c[2]'c[2] + (0.9 * c[1]'c[2] + 0.6 * c[1]'c[2]' + hc) +
        (0.4 * c[1]'c[1] * F + 0.15im * c[1]'c[2]' * F + 0.1 * c[2] * c[1] * F^2 + hc) + cnumber
    Hbdg = hilbert_space(c, 1:2, BdG())
    H = tensor_product(Hbdg, Hq)
    @test dim(H) == 4 * (2nmax + 1)
    S = representation(K, H, :dense)
    @test S ≈ representation(K, H)
    @test contained(central(2 * eigvals(Hermitian(S)), 4), manybody_differences(hilbert_space(c, 1:2), K))
    C = kron(P, [0I I(2); I(2) 0I])
    @test -C * transpose(S) * C ≈ S
    # harmonics beyond nmax still act inside the truncated space
    @test norm(representation(c[1]'c[2] * F^(2nmax) + hc, H)) > 0

    Ksp = ω * N + 0.3 * c[1]'c[1] + (0.9 * c[1]'c[2] + hc) + (0.5 * c[1]'c[1] * F + 0.2im * c[1]'c[2] * F^2 + hc) + cnumber
    Ssp = representation(Ksp, tensor_product(hilbert_space(c, 1:2, SingleParticle()), Hq), :dense)
    @test contained(central(eigvals(Hermitian(Ssp)), 2), manybody_differences(hilbert_space(c, 1:2), Ksp))

    @majoranas γ
    Km = ω * N + 0.5im * γ[1] * γ[2] + 0.4im * γ[2] * γ[3] + 0.3im * γ[3] * γ[4] +
        (0.35im * γ[1] * γ[3] * F + 0.2im * γ[2] * γ[3] * F + hc) + cnumber
    Sm = representation(Km, tensor_product(hilbert_space(γ, 1:4, SingleParticle()), Hq), :dense)
    @test contained(central(2 * eigvals(Hermitian(Sm)), 4), manybody_differences(hilbert_space(γ, 1:4), Km))
    @test -kron(P, I(4)) * conj(Sm) * kron(P, I(4)) ≈ Sm

    @test iszero(representation(cnumber, H))
    @test_throws ArgumentError representation(K + N^2, H)                 # not a derivation
    @test_throws ArgumentError representation(K + c[1]'c[1] * N, H)       # not a multiplication
    @test_throws ArgumentError representation(c[1]'c[1] * c[2]'c[2], H)   # not quadratic
end
