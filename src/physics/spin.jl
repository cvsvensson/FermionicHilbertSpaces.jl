_check_spin(J) = (J isa Real && ishalfinteger(J) && J >= 0) ? J : throw(ArgumentError("Spin J must be a nonnegative half-integer such as 1//2 or 1, got $(repr(J))."))
_to_halfint(::Nothing) = nothing
_to_halfint(spin) = HalfInt(_check_spin(spin))

# `shift` is the change of `m` per power, 1 for `:+`, -1 for `:-` and 0 for `:z`, raised to the power `exponent`.
struct Spin
    shift::Int
    exponent::Int
end
const SpinSym{M} = Op{Spin, M}
const SpinMode = Mode{<:Field{Spin}}
_spin(m::SpinMode) = m.field.params
Base.getindex(s::SpinMode, op) = _spin_operator(s, op)
Base.getindex(s::SpinMode, ::Colon) = [s[σ] for σ in (:x, :y, :z)]

"""
    @spin s

    @spin s J

Create a symbolic spin `s` for constructing spin operators such as
`s[:x]`, `s[:y]`, `s[:z]`, `s[:+]`, and `s[:-]`. The spin `J` can be given here or later to
`hilbert_space`.

The operators are spin operators, not Pauli matrices: `s[:z]` has eigenvalues `-J, …, J`, so
for spin 1/2 `s[:x]` is `σˣ/2`, and `s[:±] = s[:x] ± i s[:y]`. The basis states of
`hilbert_space(s, J)` are ordered by increasing `s[:z]` eigenvalue, starting at `-J`.

For `J ≥ 1`, `==` and `iszero` on symbolic expressions compare their stored form, which is not
unique, so equal operators can compare unequal: `iszero(s[:z]*s[:+]^2 - s[:+]^2)` is `false` for
`J = 1`.
"""
macro spin(x, spin = nothing)
    return Expr(
        :block, :($(esc(x)) = Mode(_spin_field($(Expr(:quote, x)), $(esc(spin))), ())),
        :($(esc(x))))
end

"""
    @spins s 

    @spins s J

Create a spin field `s`. Index into it to get the spin on a site:
`s[1][:z]` gives the z-operator on site 1. Operators and basis states follow the
conventions of [`@spin`](@ref).
"""
macro spins(name, spin = nothing)
    return Expr(:block, :($(esc(name)) = _spin_field($(Expr(:quote, name)), $(esc(spin)))), :($(esc(name))))
end
_spin_field(name, J) = Field{Spin}(name, nothing, _to_halfint(J))


"""
    SpinState(m)

The basis state of a single spin with `s[:z]` eigenvalue `m`, e.g. `SpinState(-1//2)`.
"""
struct SpinState{T<:Integer} <: AbstractBasisState
    m::Half{T}
end
SpinState(m) = SpinState(HalfInteger(m))
Base.:(==)(a::SpinState, b::SpinState) = a.m == b.m
Base.isless(a::SpinState, b::SpinState) = a.m < b.m
Base.hash(x::SpinState, h::UInt) = hash(twice(x.m), h)

_no_spin_count() = throw(ArgumentError("Spins have no particle number or fermion parity, so `NumberConservation` and `ParityConservation` do not apply to spin spaces. To count spin excitations, constrain an operator, such as `sum(S[i][:z] + J for i in labels) => k`."))
particle_number(::SpinState) = _no_spin_count()
parity(::SpinState) = _no_spin_count()

internal_rep(state::SpinState, ::AbstractHilbertSpace, ::Type{T}) where {T <: Integer} = T(twice(state.m))
physical_rep(state::Integer, ::Type{SpinState}) = SpinState(half(state))
function physical_rep(state::Integer, ::Type{SpinState{T}}) where {T <: Integer}
    SpinState{T}(convert(Half{T}, half(state)))
end
struct SpinSpace{T<:Integer,S} <: AbstractAtomicHilbertSpace{SpinState{T}}
    spin::Half{T}
    mode::S
end
function SpinSpace(sym::S, J) where {S <: SpinMode}
    Jhalf = HalfInteger(_check_spin(J))
    T = typeof(twice(Jhalf))
    return SpinSpace{T, S}(Jhalf, sym)
end
SpinSpace(sym::SpinMode) = (_spin(sym) isa Nothing ? throw(ArgumentError("$sym has no spin value. Pass J to hilbert_space, e.g. `hilbert_space(s, 1//2)` or `hilbert_space(s, labels, 1//2)`, or set it when creating the spin with `@spin s 1//2` or `@spins s 1//2`.")) : SpinSpace(sym, _spin(sym)))
basisstates(H::SpinSpace) = BasisStates(H, Iterators.map(n -> basisstate(n, H), 1:dim(H)))
Base.@propagate_inbounds function basisstate(n::Integer, H::SpinSpace{T}) where {T}
    @boundscheck 1 <= n <= dim(H) || throw(BoundsError(H, n))
    return SpinState{T}(n - 1 - H.spin)
end
dim(H::SpinSpace) = Int(twice(H.spin)) + 1
function _state_index(s::SpinState, space::SpinSpace)
    t = twice(s.m) + twice(space.spin) # 2(m + J)
    i = (t >> 1) + one(t) # (checking i against dim rather than t against 4J avoids slow SLP vectorization in products of spins)
    (iseven(t) && 1 <= i <= dim(space)) ? i : zero(t)
end
has_arithmetic_state_index(::SpinSpace) = true

atomic_id(H::SpinSpace) = H.mode
function add_tag(H::SpinSpace{T,S}, tag) where {T,S}
    newsym = add_tag(H.mode, tag)
    return SpinSpace{T, typeof(newsym)}(H.spin, newsym)
end
_check_spin_unset(f::Field) = f.params isa Nothing || throw(ArgumentError("The spin value was already given when creating `$f`, so `J` must be omitted."))
hilbert_space(sym::SpinMode) = SpinSpace(sym)
function hilbert_space(sym::SpinMode, J)
    _check_spin_unset(sym.field)
    SpinSpace(sym, J)
end
hilbert_space(f::Field{Spin}, labels::AbstractVector, @nospecialize(constraint::ConstraintLike = NoSymmetry())) = _mode_product(f, labels, constraint)
Base.:(==)(a::SpinSpace, b::SpinSpace) = a === b || (a.mode == b.mode && a.spin == b.spin)
Base.hash(x::SpinSpace, h::UInt) = hash(x.mode, hash(x.spin, h))

function interpret_state(input::AbstractString, H::SpinSpace{T}) where {T}
    Jhalf = H.spin
    input in ("up", "↑", "down", "↓") && Jhalf != 1 // 2 && throw(ArgumentError("The spin state `\"$input\"` is only defined for `J` = 1/2, got `J` = $Jhalf. Give `m` as a half-integer instead."))
    if input in ("up", "↑")
        return SpinState(Jhalf)
    elseif input in ("down", "↓")
        return SpinState(-Jhalf)
    end
    m = try
        parse(HalfInt, input)
    catch
        throw(ArgumentError("A spin state string must be `\"up\"`, `\"down\"` or a half-integer, got `\"$input\"`."))
    end
    m = convert(Half{T}, m)
    if m < -Jhalf || m > Jhalf
        throw(ArgumentError("The spin state `m` must be between `-J` and `J`, got `m` = $m for `J` = $Jhalf."))
    end
    return SpinState(m)
end
interpret_state(input::AbstractChar, H::SpinSpace) = interpret_state(string(input), H)

function operators(H::SpinSpace)
    Jhalf = H.spin
    Splus = spzeros(Float64, dim(H), dim(H))
    Sminus = spzeros(Float64, dim(H), dim(H))
    Sz = spzeros(Float64, dim(H), dim(H))
    for state in basisstates(H)
        m = state.m
        i = state_index(state, H)
        if m < Jhalf
            j = state_index(SpinState(m + 1), H)
            Splus[j, i] = sqrt(Jhalf * (Jhalf + 1) - m * (m + 1))
        end
        if m > -Jhalf
            j = state_index(SpinState(m - 1), H)
            Sminus[j, i] = sqrt(Jhalf * (Jhalf + 1) - m * (m - 1))
        end
        Sz[i, i] = m
    end
    return Dict(:+ => Splus, :- => Sminus, :Z => Sz, :X => (Splus + Sminus) / 2, :Y => (Splus - Sminus) / (2im))
end

@testitem "Spin" begin
    using FermionicHilbertSpaces: SpinState, operators, Half
    using LinearAlgebra
    @spin s
    @test basisstates(hilbert_space(s, 1 // 2)) == [SpinState(-1 // 2), SpinState(1 // 2)]
    @test basisstates(hilbert_space(s, 1)) == [SpinState(-1), SpinState(0), SpinState(1)]
    # equal states hash equally, also across integer types
    s16 = SpinState{Int16}(Half{Int16}(3 // 2))
    @test s16 == SpinState(3 // 2) && hash(s16) == hash(SpinState(3 // 2))
    @test hash(SpinState(1 // 2)) != hash(SpinState(-1 // 2))

    H = hilbert_space(s, 1 // 2)
    S = operators(H)
    @test S[:+] == [0 0; 1 0]
    @test S[:-] == [0 1; 0 0]
    @test S[:Z] == [-1//2 0; 0 1//2]
    @test S[:X] == [0 1//2; 1//2 0]
    @test S[:Y] == [0 im//2; -im//2 0]
    # test pauli algebra
    @test S[:X] * S[:Y] - S[:Y] * S[:X] ≈ im * S[:Z]
    @test S[:Y] * S[:Z] - S[:Z] * S[:Y] ≈ im * S[:X]
    @test S[:Z] * S[:X] - S[:X] * S[:Z] ≈ im * S[:Y]
    # spin algebra for higher spins
    for j in (1, 3 // 2, 2)
        Sj = operators(hilbert_space(s, j))
        @test all(Sj[a] * Sj[b] - Sj[b] * Sj[a] ≈ im * Sj[c] for (a, b, c) in ((:X, :Y, :Z), (:Y, :Z, :X), (:Z, :X, :Y)))
        @test Sj[:X]^2 + Sj[:Y]^2 + Sj[:Z]^2 ≈ j * (j + 1) * I
    end
    S1 = operators(hilbert_space(s, 1))
    @test S1[:Z] == [-1 0 0; 0 0 0; 0 0 1]
    @test S1[:+] ≈ [0 0 0; √2 0 0; 0 √2 0]
    # states outside the space have index 0
    @test [state_index(SpinState(m), H) for m in -3//2:1//2:3//2] == [0, 0, 1, 0, 2, 0, 0]
    @test [state_index(SpinState(m), hilbert_space(s, 1)) for m in -2:1//2:2] == [0, 0, 1, 0, 2, 0, 3, 0, 0]
    @spins k
    H1, H2 = [hilbert_space(k[i], 1 // 2) for i in 1:2]
    P = tensor_product(H1, H2)
    @test partial_trace(1.0 * I(dim(P)), P => H1) ≈ dim(H2) * I(dim(H1))

    ops1 = operators(H1)
    @test all(partial_trace(embed(op, H1 => P), P => H1) ≈ dim(H2) * op for op in values(ops1))

    # Invalid or missing spin values
    @spins t
    for J in (-1 // 2, 1 // 3, 0.3, "1/2", nothing)
        @test_throws ArgumentError hilbert_space(s, J)
        @test_throws ArgumentError hilbert_space(t, 1:2, J)
    end
    @test_throws ArgumentError (@spins x -1)
    @test_throws "has no spin value" hilbert_space(t, 1:2)
    @test_throws "has no spin value" hilbert_space(s)
    @test dim(hilbert_space(s, 0)) == 1

    # every spin macro form returns the created spin or field
    @test (@spins u 1 // 2) === u
    @test (@spins v) === v
    @test (@spin w 1) === w

    @fermions f
    Hf = hilbert_space(f, 1:2)
    mf = rand(dim(Hf), dim(Hf))
    Pf = tensor_product(Hf, P)
    @test partial_trace(embed(mf, Hf => Pf), Pf => Hf) ≈ dim(P) * mf

    mf1 = rand(2, 2)
    Hf1 = hilbert_space(f, 1:1)
    @test partial_trace(embed(mf1, Hf1 => Pf), Pf => Hf) ≈ dim(P) * embed(mf1, Hf1 => Hf)

end

# The spin operator `op` on `mode`: `:x`, `:y`, `:z`, `:+`, `:-`, an alias of these, or the identity `:I`.
function _spin_operator(mode, op)
    if op in _spin_x_aliases
        return Op(Spin(1, 1), mode) / 2 + Op(Spin(-1, 1), mode) / 2
    elseif op in _spin_y_aliases
        return Op(Spin(1, 1), mode) / (2im) - Op(Spin(-1, 1), mode) / (2im)
    elseif op === :I
        return 1 // 1
    else
        return _spin_power(mode, _spin_shift(op), 1)
    end
end
# The power `exponent` of the operator with the given `shift`, which is a number when the power is the identity or zero.
function _spin_power(mode, shift, exponent)
    exponent == 0 && return 1 // 1
    !(_spin(mode) isa Nothing) && shift != 0 && exponent > Int(2 * _spin(mode)) && return 0 // 1
    return Op(Spin(shift, exponent), mode)
end
const _spin_x_aliases = (:x, :X)
const _spin_y_aliases = (:y, :Y)
const _spin_z_aliases = (:z, :Z)
const _spin_plus_aliases = (:+, :plus, :p)
const _spin_minus_aliases = (:-, :minus, :m)
function _spin_shift(op)
    if op in _spin_z_aliases
        0
    elseif op in _spin_plus_aliases
        1
    elseif op in _spin_minus_aliases
        -1
    else
        hint = op isa Integer ? " `@spin S` is a single spin with operators such as `S[:z]`, and `@spins s` gives the sites `s[i]`." : ""
        throw(ArgumentError("Invalid spin operator symbol, got `$(repr(op))`.$hint"))
    end
end

function Base.show(io::IO, x::SpinSym)
    print(io, x.mode, "[:", (:-, :z, :+)[x.kind.shift + 2], "]")
    return if x.kind.exponent != 1
        print(io, "^", x.kind.exponent)
    end
end

Base.adjoint(k::Spin) = Spin(-k.shift, k.exponent)

function _spin_commutator_term(a::SpinSym, b::SpinSym)
    ka, kb = a.kind, b.kind
    if ka.shift != 0 && kb.shift == 0
        return (-ka.shift, Op(Spin(ka.shift, 1), a.mode))
    elseif ka.shift == -1 && kb.shift == 1
        return (-2, Op(Spin(0, 1), a.mode))
    else
        return nothing
    end
end

# Equal modes can have different types, such as labels `1` and `Int32(1)`.
local_mul_effect(a::SpinSym, b::SpinSym) = local_mul_effect(a, Op(b.kind, a.mode))
function local_mul_effect(a::S, b::S) where {S <: SpinSym}
    ka, kb = a.kind, b.kind
    if ka.shift == kb.shift
        exponent = ka.exponent + kb.exponent
        if ka.shift == 0
            if _spin(a.mode) == 1 // 2
                div, rem = divrem(exponent, 2)
                return (1 // 4)^div * _spin_power(a.mode, 0, rem)
            elseif _spin(a.mode) == 1
                return Op(Spin(0, iseven(exponent) ? 2 : 1), a.mode)
            end
        end
        return _spin_power(a.mode, ka.shift, exponent)
    end

    # Spin-aware identity: S+ S- = J(J+1) - Sz^2 + Sz
    if ka.shift == 1 && kb.shift == -1 && !(_spin(a.mode) isa Nothing)
        J = _spin(a.mode)
        left = Op(Spin(1, ka.exponent - 1), a.mode)
        right = Op(Spin(-1, kb.exponent - 1), a.mode)

        left_factors = ka.exponent == 1 ? Any[] : [left]
        right_factors = kb.exponent == 1 ? Any[] : [right]

        const_term = J * (J + 1)
        term_const = if iszero(const_term)
            0
        elseif length(left_factors) == 0 && length(right_factors) == 0
            const_term
        else
            NCMul(const_term, S[left_factors..., right_factors...])
        end
        term_sz2 = J == 1 // 2 ? NCMul(-1 // 4, S[left_factors..., right_factors...]) : NCMul(-1 // 1, S[left_factors..., Op(Spin(0, 2), a.mode), right_factors...])
        term_sz = NCMul(1 // 1, S[left_factors..., Op(Spin(0, 1), a.mode), right_factors...])
        return AddTerms((term_const, term_sz2, term_sz))
    end

    # Sz^n S± = (±1/2)^n S± for spin 1/2
    _spin(a.mode) == 1 // 2 && ka.shift == 0 && return NCMul((kb.shift // 2)^ka.exponent, S[b])

    comm = _spin_commutator_term(a, b)
    if isnothing(comm)
        return nothing
    end

    comm_coeff, comm_middle_factor = comm

    # For A^N B^M (out-of-order), rewrite only the middle pair:
    # A^(N-1) * (B*A + [A,B]) * B^(M-1)
    left_factors = ka.exponent == 1 ? S[] : S[Op(Spin(ka.shift, ka.exponent - 1), a.mode)]
    right_factors = kb.exponent == 1 ? S[] : S[Op(Spin(kb.shift, kb.exponent - 1), a.mode)]
    swapped_factors = S[left_factors..., Op(Spin(kb.shift, 1), a.mode), Op(Spin(ka.shift, 1), a.mode), right_factors...]
    comm_factors = S[left_factors..., comm_middle_factor, right_factors...]

    swapped_term = NCMul(1 // 1, swapped_factors)
    comm_term = NCMul(comm_coeff // 1, comm_factors)
    return AddTerms((swapped_term, comm_term))
end

"""
    apply_local_operator(op::Spin, state::SpinState, space::SpinSpace, precomp) -> (newstate, amplitude)

Apply a single spin operator to a spin state. Returns (newstate, amplitude) where amplitude is 0 if the operation is not allowed, and a nonzero value otherwise.
"""
function apply_local_operator(op::Spin, state::SpinState, space::SpinSpace, precomp, ::Type{T} = Float64) where {T}
    J = space.spin
    m = state.m
    shift = op.shift
    # S_z |m⟩ = m |m⟩
    iszero(shift) && return (state, _fast_spin_z_exponent(T, m, op.exponent))
    # S_± |m⟩ = √(J(J+1) - m(m±1)) |m±1⟩
    newm = m + shift * op.exponent
    abs(newm) > J && return (state, zero(T))
    return (SpinState(newm), _fast_spin_prefactor(T, J, m, shift, op.exponent))
end

function _fast_spin_z_exponent(::Type{T}, m::HalfInteger, exp::Int) where {T}
    exp == 0 && return one(T)
    m == 1 && return one(T)
    m == -1 && return isodd(exp) ? -one(T) : one(T)
    x = T(twice(m)) / 2
    y = x
    for _ in Base.OneTo(exp - 1)
        y *= x
    end
    return y
end
function _fast_spin_prefactor(::Type{T}, J::HalfInteger, m::HalfInteger, sign::Int, exp::Int) where {T}
    x = one(T)
    J2 = twice(J)
    m2 = twice(m)
    for _dm in Base.OneTo(exp)
        dm = _dm - 1
        x *= sqrt(T(J2 * (J2 + 2) - (m2 + dm * 2sign) * (m2 + _dm * 2sign))) / 2
    end
    return x
end

@testitem "Spin: no particle number or parity" begin
    @spin S 1//2
    @boson b
    H = tensor_product(hilbert_space(S), hilbert_space(b, 3))
    @test_throws ArgumentError constrain_space(H, NumberConservation(1))
    @test_throws ArgumentError constrain_space(H, ParityConservation(1))
end

@testitem "SpinSym" begin
    @spin S1
    @spin S2
    @fermions f
    @majoranas γ
    @boson b

    Sz1 = S1[:z]
    Sp1 = S1[:+]
    Sm1 = S1[:-]
    Sx1 = S1[:x]
    @test_throws ArgumentError S1[1]
    Sy1 = S1[:y]

    Sz2 = S2[:z]
    Sp2 = S2[:+]
    Sm2 = S2[:-]

    @test 1 * Sz1 == Sz1
    @test 1 * Sz1 + 0 == Sz1
    @test hash(Sz1) == hash(1 * Sz1) == hash(1 * Sz1 + 0)

    # Test canonical commutation relations
    # [Sz, S+] = S+
    @test Sz1 * Sp1 - Sp1 * Sz1 == Sp1
    # [Sz, S-] = -S-
    @test Sz1 * Sm1 - Sm1 * Sz1 == -Sm1
    # [S+, S-] = 2Sz
    @test Sp1 * Sm1 - Sm1 * Sp1 == 2 * Sz1

    # Reordering powers: Sz S± = S± (Sz ± 1)
    @test Sp1^2 * Sz1 == Sz1 * Sp1^2 - 2 * Sp1^2
    @test Sm1^2 * Sz1 == Sz1 * Sm1^2 + 2 * Sm1^2
    @test Sp1 * Sz1^2 == (Sz1 - 1)^2 * Sp1
    @test Sm1^2 * Sz1^2 == (Sz1 + 2)^2 * Sm1^2
    @test Sm1^2 * Sp1 - Sp1 * Sm1^2 == -2 * (Sz1 * Sm1 + Sm1 * Sz1)
    @test (Sp1^2 * Sz1) * Sm1 == Sp1^2 * (Sz1 * Sm1)

    # Test x and y operators expand correctly
    @test Sx1 == (Sp1 + Sm1) / 2
    @test Sy1 == (Sp1 - Sm1) / (2im)

    # Spins at different sites commute
    @test Sz1 * Sz2 - Sz2 * Sz1 == 0
    @test Sp1 * Sp2 - Sp2 * Sp1 == 0
    @test Sp1 * Sm2 - Sm2 * Sp1 == 0

    # Spins commute with fermions
    @test Sz1 * f[1] - f[1] * Sz1 == 0
    @test Sp1 * f[1] - f[1] * Sp1 == 0
    @test Sz1 * f[1]' - f[1]' * Sz1 == 0
    @test Sp1 * f[1]' - f[1]' * Sp1 == 0

    # Spins commute with majoranas
    @test Sz1 * γ[1] - γ[1] * Sz1 == 0
    @test Sp1 * γ[1] - γ[1] * Sp1 == 0

    # Spins commute with bosons
    @test Sz1 * b - b * Sz1 == 0
    @test Sp1 * b - b * Sp1 == 0
    @test Sz1 * b' - b' * Sz1 == 0
    @test Sp1 * b' - b' * Sp1 == 0

    # Test adjoint
    @test Sp1' == Sm1
    @test Sm1' == Sp1
    @test Sz1' == Sz1

    @test iszero(Sz1 - Sz1)
    @test iszero(2 * Sz1 - 2 * Sz1)
    @test iszero(0 * Sz1)
    @test iszero(Sz1 * 0)
    @test iszero(0 * (Sp1 + Sm1))
    @test iszero((Sp1 + Sm1) * 0)

    # Test algebraic properties
    @test 1 + (Sp1 + Sm1) == 1 + Sp1 + Sm1 == Sp1 + Sm1 + 1 == Sp1 + 1 + Sm1

    # Test with explicit spin
    @spin Shalf 1 // 2
    @test iszero(Shalf[:+]^2)
    @test iszero(Shalf[:-]^2)
    @test Shalf[:+] * Shalf[:-] == Shalf[:z] + 1 // 2

    @spin Sone 1
    Spo = Sone[:+]
    @test iszero(Sone[:+]^3)
    @test Sone[:z]^3 == Sone[:z]

    # Test spin-1/2 identity: Sx^2 + Sy^2 + Sz^2 = 3/4 * I, and Colon operator
    @test Shalf[:]' * Shalf[:] == 3 / 4 * Shalf[:I]

    # Equal bases of different types
    @spins s 1 // 2
    @test s[1][:z] * s[Int32(1)][:z] == s[1][:z]^2
end

@testitem "Spin matrix reps" begin
    @spin s
    H = hilbert_space(s, 1 // 2)
    S_mat = Dict(sym => representation(s[sym], H) for sym in (:x, :y, :z, :+, :-))
    @test S_mat[:+] == [0 0; 1 0]
    @test S_mat[:-] == [0 1; 0 0]
    @test S_mat[:z] == [-1//2 0; 0 1//2]
    @test S_mat[:x] == [0 1//2; 1//2 0]
    @test S_mat[:y] == [0 im//2; -im//2 0]
end

@testitem "Heisenberg dimer" begin
    using LinearAlgebra
    @spins s 1 // 2
    H = hilbert_space(s, 1:2)
    dot_spins(a, b) = sum(a[c] * b[c] for c in (:x, :y, :z))
    # singlet at -3/4, triplet at 1/4
    @test eigvals(Hermitian(representation(dot_spins(s[1], s[2]), H, Matrix))) ≈ [-3 / 4, 1 / 4, 1 / 4, 1 / 4]
    Stot = (s[1][c] + s[2][c] for c in (:x, :y, :z))
    @test eigvals(Hermitian(representation(sum(S^2 for S in Stot), H, Matrix))) ≈ [0, 2, 2, 2]
end

@testitem "Spin: spin-aware vs not" begin
    for J in (1 // 2, 1, 3 // 2)
        @spin s J
        H = hilbert_space(s)
        @test dim(H) == 2J + 1
        @test iszero(s[:+]^Int(2J + 1))
        @test iszero(s[:-]^Int(2J + 1))
        syms = (:+, :-, :z, :x, :y, :I)
        smat = Dict(sym => representation(s[sym], H) for sym in syms)
        @test all(smat[sym1] * smat[sym2] ≈ representation(s[sym1] * s[sym2], H) for sym1 in syms for sym2 in syms)
        # triple products exercise reordering of powers
        @test all(smat[s1] * smat[s2] * smat[s3] ≈ representation(s[s1] * s[s2] * s[s3], H) for s1 in syms for s2 in syms for s3 in syms)
    end
end
