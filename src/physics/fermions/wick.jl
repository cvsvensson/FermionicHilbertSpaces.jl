# Each term of `op` is expanded on its own, so that the factors of every expansion have a single concrete type.
# A sum is traversed through its dictionary, whose entries have a concrete type, unlike those of `termpairs`.
sum_over_terms(f, op::NCAdd) = mapreduce(((k, c),) -> f(k.factors, c), +, op.dict; init = f((), op.coeff))
sum_over_terms(f, op) = mapreduce(((fs, c),) -> f(fs, c), +, NonCommutativeProducts.termpairs(1 * op); init = 0)

# Wick's theorem for a state with vanishing one-point function: c * prod(fs) is the sum over pairings of some factors
# of the products of their contractions `contraction(x, y) = ⟨xy⟩` times the normal ordered product :free…: of the
# others, with the phase of moving the pairs to the front. Returns the sum of `f(coeff, free)` over the terms with at
# most `maxfree` uncontracted factors.
function wick_sum(f, c, fs, contraction, maxfree, free = fs[1:0])
    isempty(fs) && return f(c, free)
    x, rest = fs[1], fs[2:end]
    s = length(free) < maxfree ? wick_sum(f, c, rest, contraction, maxfree, [free; x]) : 0
    for j in eachindex(rest)
        y = rest[j]
        λ = _phase(free, x) * _phase(free, y) * _phase(view(rest, 1:(j - 1)), y)
        s += wick_sum(f, λ * c * contraction(x, y), deleteat!(copy(rest), j), contraction, maxfree, free)
    end
    return s
end
_phase(xs, y) = prod(x -> exchange_phase(x, y), xs; init = 1)

# The coefficients are multiplied by `unit`, the element type of the correlation matrix, before the expansion, so that
# operators with different coefficient types share its compiled code.
function wick_expectation(op, contraction, keep = Returns(true), unit = 1)
    return sum_over_terms(op) do fs, c
        keep(fs) || return zero(c)
        return wick_sum((d, _) -> d, c * unit, fs, contraction, 0)
    end
end
function _mean_field(op, contraction, keep = Returns(true), unit = 1)
    return sum_over_terms(op) do fs, c
        keep(fs) || return 0
        c *= unit
        length(fs) <= 2 && return c * prod(fs; init = 1)
        return wick_sum(c, fs, contraction, 2) do d, xs
            length(xs) == 2 ? d * (xs[1] * xs[2] - contraction(xs[1], xs[2])) : d * prod(xs; init = 1)
        end
    end
end

function _mode_position(x, H)
    position = x isa Union{FermionSym, MajoranaSym} ? mode_position(x, H) : 0
    iszero(position) && throw(ArgumentError("The operator `$x` is not a mode of the space `$H`."))
    return position
end
_nambu_index(x, H) = state_index(NambuState(_mode_position(x, H), !x.creation), H)
_contraction(G, H::BdGHilbertSpace) = (x, y) -> G[_nambu_index(x', H), _nambu_index(y, H)]
_contraction(G, H::MajoranaSingleParticleHilbertSpace) = (x, y) -> 2 * G[_mode_position(x, H), _mode_position(y, H)]
function _contraction(G, H::SingleParticleHilbertSpace)
    index(x) = state_index(SingleParticleState(_mode_position(x, H)), H)
    return function (x, y)
        i, j = index(x), index(y)
        return x.creation == y.creation ? zero(eltype(G)) : x.creation ? G[i, j] : (i == j) - G[j, i]
    end
end
_acts_within(fs, ::SingleParticleHilbertSpace) = 2 * count(x -> x.creation, fs) == length(fs)
_acts_within(fs, ::Union{BdGHilbertSpace, MajoranaSingleParticleHilbertSpace}) = iseven(length(fs))
# The modes are checked for every term, since not every term looks up all of its modes in `G`.
function _in_space(fs, H, projection)
    foreach(x -> _mode_position(x, H), fs)
    _acts_within(fs, H) && return true
    projection || throw(ArgumentError("The term `$(prod(fs))` does not act within the space `$H`. Pass `projection=true` to drop such terms."))
    return false
end
function expectation(op, G::AbstractMatrix, H::Union{SingleParticleHilbertSpace, BdGHilbertSpace, MajoranaSingleParticleHilbertSpace}; projection = false)
    return zero(eltype(G)) + wick_expectation(op, _contraction(G, H), fs -> _in_space(fs, H, projection), one(eltype(G)))
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
mean_field(op, G::AbstractMatrix, H::Union{SingleParticleHilbertSpace, BdGHilbertSpace, MajoranaSingleParticleHilbertSpace}; projection = false) = _mean_field(op, _contraction(G, H), fs -> _in_space(fs, H, projection), one(eltype(G)))

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
