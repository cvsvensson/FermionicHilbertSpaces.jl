# Normal ordering relative to a Gaussian state with vanishing one-point function. `Unordered(x)` is a factor that is
# not yet absorbed and `Fluctuation(x)` a factor of the product :…: that is normal ordered relative to the state, so
# that x :y…: = λ :y x…: + ⟨xy⟩ :…:. Products end in `OrderEnd`, where the last unordered factor x becomes :x:.
struct Unordered{X, C}
    x::X
    contraction::C
end
struct Fluctuation{X}
    x::X
end
struct OrderEnd end
@nc Unordered Fluctuation OrderEnd
Base.:(==)(a::Fluctuation, b::Fluctuation) = a.x == b.x
Base.hash(a::Fluctuation, h::UInt) = hash(a.x, hash(Fluctuation, h))

NonCommutativeProducts.mul_effect(a::Unordered, b::Fluctuation) = AddTerms((Swap(exchange_phase(a.x, b.x)), a.contraction(a.x, b.x)))
NonCommutativeProducts.mul_effect(a::Unordered, ::OrderEnd) = Fluctuation(a.x) * OrderEnd()
# :xx: = 0 for fermions and Majoranas alike
NonCommutativeProducts.mul_effect(a::Fluctuation, b::Fluctuation) = a == b ? 0 : isless(b.x, a.x) ? Swap(exchange_phase(a.x, b.x)) : nothing
NonCommutativeProducts.mul_effect(::Union{Unordered, Fluctuation, OrderEnd}, ::Union{Unordered, Fluctuation, OrderEnd}) = nothing

# Each term of `op` is expanded on its own, since the expansion of the whole sum is much larger than what is kept of it.
# A sum is traversed through its dictionary, whose entries have a concrete type, unlike those of `termpairs`.
sum_over_terms(f, op::NCAdd) = mapreduce(((k, c),) -> f(k.factors, c), +, op.dict; init = f((), op.coeff))
sum_over_terms(f, op) = mapreduce(((fs, c),) -> f(fs, c), +, NonCommutativeProducts.termpairs(1 * op); init = 0)
# The product of `fs` as a sum of products of `Fluctuation`s, each followed by `OrderEnd`, with `contraction(x, y) = ⟨xy⟩`.
# The sorter applies the Wick rules, so it must run even when autosort is disabled.
state_normal_order(fs, contraction) = NonCommutativeProducts.with_autosort(() -> prod(x -> Unordered(x, contraction), fs; init = 1) * OrderEnd(), true)
fluctuations(fs) = [f.x for f in fs if f isa Fluctuation]

function wick_expectation(op, contraction, keep = Returns(true))
    return sum_over_terms(op) do fs, c
        keep(fs) || return zero(c)
        length(fs) == 2 && return c * contraction(fs[1], fs[2])
        terms = NonCommutativeProducts.termpairs(state_normal_order(fs, contraction))
        return c * sum((d for (gs, d) in terms if isempty(fluctuations(gs))); init = 0)
    end
end
function _mean_field(op, contraction, keep = Returns(true))
    return sum_over_terms(op) do fs, c
        keep(fs) || return 0
        length(fs) <= 2 && return c * prod(fs; init = 1)
        return c * sum_over_terms(state_normal_order(fs, contraction)) do gs, d
            xs = fluctuations(gs)
            length(xs) > 2 ? 0 : d * (prod(xs; init = 1) - (length(xs) == 2 ? contraction(xs...) : 0))
        end
    end
end

function _mode_position(x, H)
    position = x isa FermionSym ? mode_position(x, H) : 0
    iszero(position) && throw(ArgumentError("The operator `$x` is not a fermion of the space `$H`."))
    return position
end
_nambu_index(x, H) = state_index(NambuState(_mode_position(x, H), !x.creation), H)
_contraction(G, H::BdGHilbertSpace) = (x, y) -> G[_nambu_index(x', H), _nambu_index(y, H)]
function _contraction(G, H::SingleParticleHilbertSpace)
    index(x) = state_index(SingleParticleState(_mode_position(x, H)), H)
    return function (x, y)
        i, j = index(x), index(y)
        return x.creation == y.creation ? zero(eltype(G)) : x.creation ? G[i, j] : (i == j) - G[j, i]
    end
end
_acts_within(fs, ::SingleParticleHilbertSpace) = 2 * count(x -> x.creation, fs) == length(fs)
_acts_within(fs, ::BdGHilbertSpace) = iseven(length(fs))
# The modes are checked for every term, since not every term looks up all of its modes in `G`.
function _in_space(fs, H, projection)
    foreach(x -> _mode_position(x, H), fs)
    _acts_within(fs, H) && return true
    projection || throw(ArgumentError("The term `$(prod(fs))` does not act within the space `$H`. Pass `projection=true` to drop such terms."))
    return false
end
function expectation(op, G::AbstractMatrix, H::Union{SingleParticleHilbertSpace, BdGHilbertSpace}; projection = false)
    return zero(eltype(G)) + wick_expectation(op, _contraction(G, H), fs -> _in_space(fs, H, projection))
end

"""
    mean_field(op, G, H; projection = false)

Return the mean-field operator of `op` in the Gaussian state with correlation matrix `G` on the single-particle or BdG
space `H` (see [`expectation`](@ref)). Every product in `op` is replaced by its part of order at most two in the
fluctuations `x - ⟨x⟩` (the Hartree, Fock and pairing terms), so the result is of degree at most two and has the
same expectation value as `op` in the state. `projection` is as in [`expectation`](@ref).

```julia
@fermions f
H = hilbert_space(f, 1:2, SingleParticle())
mean_field(f[1]' * f[1] * f[2]' * f[2], [0.5 0; 0 0.25], H) # 0.25 f[1]' f[1] + 0.5 f[2]' f[2] - 0.125
```
"""
mean_field(op, G::AbstractMatrix, H::Union{SingleParticleHilbertSpace, BdGHilbertSpace}; projection = false) = _mean_field(op, _contraction(G, H), fs -> _in_space(fs, H, projection))

@testitem "Wick expectation values" begin
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra
    @fermions f
    N, β = 3, 0.7
    Hfock = hilbert_space(f, 1:N)
    thermal(q) = (ρ = exp(-β * representation(q, Hfock, :dense)); ρ / tr(ρ))
    hop = sum(randn(ComplexF64) * f[i]' * f[j] for i in 1:N, j in 1:N)
    pair = sum(randn(ComplexF64) * f[i]' * f[j]' for i in 1:N, j in 1:N)
    quartic = sum(randn(ComplexF64) * f[i]' * f[j]' * f[k] * f[l] for i in 1:N, j in 1:N, k in 1:N, l in 1:N)
    sextic = f[1]' * f[2]' * f[3]' * f[3] * f[1] * f[2] + 2 * f[2]' * f[1] * f[3]' * f[2] * f[1]' * f[3]
    pairing_ops = (quartic + f[1]' * f[2]' * f[3]' * f[1], sextic + f[1] * f[2] * f[3]' * f[1]' * f[2] * f[3] + 1)

    H = hilbert_space(f, 1:N, SingleParticle())
    q = hop + hop'
    ρ = thermal(q)
    G = transpose(inv(I + exp(β * representation(q, H, :dense))))
    for op in (quartic + quartic', sextic + hc)
        @test expectation(op, G, H) ≈ tr(ρ * representation(op, Hfock, :dense))
    end

    H = hilbert_space(f, 1:N, BdG())
    q = hop + pair + hc
    ρ = thermal(q)
    G = transpose(inv(I + exp(2β * representation(q, H, :dense))))
    for op in pairing_ops
        @test expectation(op, G, H) ≈ tr(ρ * representation(op, Hfock, :dense))
    end
    unsorted_ev = FermionicHilbertSpaces.NonCommutativeProducts.with_autosort(() -> expectation(f[2] * f[1]' * f[2]' * f[1], G, H), false)
    @test unsorted_ev ≈ expectation(f[2] * f[1]' * f[2]' * f[1], G, H)

    for H in (hilbert_space(f, 1:N, SingleParticle()), hilbert_space(f, 1:N, BdG())), op in (f[99], f[1]' * f[1] + f[99])
        G0 = zeros(dim(H), dim(H))
        @test_throws ArgumentError expectation(op, G0, H)
        @test_throws ArgumentError mean_field(op, G0, H)
    end
    n = f[1]' * f[1]
    for (H, outside) in ((hilbert_space(f, 1:N, SingleParticle()), f[1]' * f[2]'), (hilbert_space(f, 1:N, BdG()), f[1]' * f[2]' * f[3]))
        G0 = zeros(dim(H), dim(H))
        @test_throws ArgumentError expectation(outside + n, G0, H)
        @test_throws ArgumentError mean_field(outside + n, G0, H)
        @test expectation(outside, G0, H; projection = true) === 0.0
        @test mean_field(outside + n, G0, H; projection = true) == n
    end
end

@testitem "Wick expectation values of Majoranas" begin
    using LinearAlgebra
    @majoranas γ
    H = hilbert_space(γ, 1:4)
    A = randn(4, 4)
    ρ = exp(-representation(1im * sum((A[i, j] - A[j, i]) * γ[i] * γ[j] for i in 1:4, j in 1:4), H, :dense))
    ρ /= tr(ρ)
    ev(op) = tr(ρ * representation(op, H, :dense))
    op = γ[1] * γ[2] * γ[3] * γ[4] + 2 * γ[4] * γ[1] * γ[3] * γ[1] + γ[2] * γ[3]
    @test FermionicHilbertSpaces.FreeFermions.wick_expectation(op, (x, y) -> ev(x * y)) ≈ ev(op)
end

@testitem "Mean field" begin
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra
    @fermions f
    N = 3
    H = hilbert_space(f, 1:N, BdG())
    q = sum(randn(ComplexF64) * f[i]' * f[j] + randn(ComplexF64) * f[i]' * f[j]' for i in 1:N, j in 1:N)
    G = transpose(inv(I + exp(2 * representation(q + q', H, :dense))))
    ev(op) = expectation(op, G, H)
    n1, n2 = f[1]' * f[1], f[2]' * f[2]
    hartree = ev(n2) * n1 + ev(n1) * n2 - ev(n1) * ev(n2)
    fock = -ev(f[1]' * f[2]) * f[2]' * f[1] - ev(f[2]' * f[1]) * f[1]' * f[2] + ev(f[1]' * f[2]) * ev(f[2]' * f[1])
    pairing = ev(f[1]' * f[2]') * f[2] * f[1] + ev(f[2] * f[1]) * f[1]' * f[2]' - ev(f[1]' * f[2]') * ev(f[2] * f[1])
    Hfock = hilbert_space(f, 1:N)
    @test representation(mean_field(n1 * n2, G, H), Hfock, :dense) ≈ representation(hartree + fock + pairing, Hfock, :dense)

    op = f[1]' * f[2]' * f[3]' * f[3] * f[1] * f[2] + f[1]' * f[2]' * f[3] * f[1] + hc
    @test ev(mean_field(op, G, H)) ≈ ev(op)
end

@testitem "Mean field with symbolic coefficients" begin
    using FermionicHilbertSpaces.FreeFermions
    using Symbolics
    @variables U
    @fermions f
    H = hilbert_space(f, 1:2, SingleParticle())
    G = [1 // 2 1 // 5; 1 // 5 1 // 4]
    op = f[1]' * f[1] * f[2]' * f[2]
    mf = mean_field(U * op, G, H)
    @test isequal(Symbolics.simplify(expectation(mf, G, H) - U * expectation(op, G, H)), 0)
end
