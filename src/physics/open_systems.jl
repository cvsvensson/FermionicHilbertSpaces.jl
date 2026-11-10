
struct TransposedSpace{B,H} <: AbstractHilbertSpace{B}
    parent::H
end
TransposedSpace(inner::H) where {H<:AbstractHilbertSpace} = TransposedSpace{statetype(inner),H}(inner)
TransposedSpace(H::TransposedSpace) = parent(H)

Base.:(==)(a::TransposedSpace, b::TransposedSpace) = a.parent == b.parent
Base.hash(H::TransposedSpace, h::UInt) = hash(H.parent, h)
basisstates(H::TransposedSpace) = basisstates(H.parent)
Base.@propagate_inbounds basisstate(i::Integer, H::TransposedSpace) = basisstate(i, H.parent)
_state_index(state, H::TransposedSpace) = _state_index(state, H.parent)
has_arithmetic_state_index(H::TransposedSpace) = has_arithmetic_state_index(H.parent)
dim(H::TransposedSpace) = dim(H.parent)
isconstrained(H::TransposedSpace) = isconstrained(H.parent)
group_id(H::TransposedSpace) = group_id(H.parent)
atomic_id(H::TransposedSpace) = atomic_id(H.parent)
atom_ids(H::TransposedSpace) = atom_ids(H.parent)
atom_position(x, H::TransposedSpace) = atom_position(x, H.parent)
mode_ordering(H::TransposedSpace) = mode_ordering(H.parent)
atomic_factors(H::TransposedSpace) = map(TransposedSpace, atomic_factors(H.parent)) # (H,)
atomic_factor(H::TransposedSpace, i) = TransposedSpace(atomic_factor(H.parent, i))
add_tag(H::TransposedSpace, tag) = TransposedSpace(add_tag(H.parent, tag))
groups(H::TransposedSpace) = map(TransposedSpace, groups(H.parent))
factors(H::TransposedSpace) = map(TransposedSpace, factors(H.parent))
TransposedSpace(H::ProductSpace) = ProductSpace(map(TransposedSpace, groups(H)))
TransposedSpace(H::AbstractSubspace) = map_parent(TransposedSpace, H)
state_mapper(H::TransposedSpace, Hs) = state_mapper(H.parent, Hs)
combine_states(states, H::TransposedSpace) = combine_states(states, H.parent)
Base.parent(H::TransposedSpace) = H.parent
partial_trace_phase_factor(f1, f2, H::TransposedSpace) = partial_trace_phase_factor(f1, f2, parent(H))
partial_trace_phase_factor_eltype(H::TransposedSpace) = partial_trace_phase_factor_eltype(parent(H))
partial_transpose_phase_factor_eltype(H::TransposedSpace) = partial_transpose_phase_factor_eltype(parent(H))
partial_transpose_phase_factor(f1, f2, H::TransposedSpace) = partial_transpose_phase_factor(f1, f2, parent(H))

function FermionicSpace(spaces::AbstractVector{<:TransposedSpace}, group::Group)
    only(unique(map(group_id, spaces))) == group || throw(ArgumentError("All spaces must belong to the group `$group`."))
    TransposedSpace(FermionicSpace(map(H -> H.parent, spaces), group))
end

combine_into_group(group::Group{Majorana}, spaces::AbstractVector{<:TransposedSpace}) =
    TransposedSpace(combine_into_group(group, map(parent, spaces)))


tag_left(base) = add_tag(base, :left)
tag_right(base) = add_tag(base, :right)

"""
    open_system(sym, args...; kwargs...)
    open_system(H; kwargs...)

Set up the doubled space for density matrices of a system with Hilbert space `H`, or of
`hilbert_space(sym, args...; kwargs...)`. For `H`, `kwargs` are passed to `tensor_product`. Returns `(Hlr, Hl, Hr, left, right)`:

- `Hl` and `Hr` are copies of the system space tagged `:left` and `:right`. `Hr` is transposed,
  so that `representation(right(A * B), Hr) == representation(right(B), Hr) * representation(right(A), Hr)`.
- `Hlr = tensor_product(Hl, Hr)` is the space of vectorized density matrices.
- `left(op)` and `right(op)` map a symbolic operator of the system to the copy acting on `Hl` or
  `Hr`, so `left(A) * right(B)` represents `ρ -> A ρ B` and `-1im * (left(H) - right(H))` is
  the Liouvillian of a Hamiltonian `H`.

Use `reshape(ρ, (Hl, Hr) => Hlr)` and `reshape(v, Hlr => (Hl, Hr))` to convert between density
matrices and vectors.

```jldoctest
julia> using FermionicHilbertSpaces: dissipator; @fermions f;

julia> Hlr, Hl, Hr, left, right = open_system(f, 1:1);

julia> n = f[1]' * f[1];

julia> L = representation(-1im * (left(n) - right(n)) + dissipator(f[1]), Hlr);

julia> size(L)
(4, 4)
```
"""
function open_system(base, args...; kwargs...)
    left_basis = tag_left(base)
    right_basis = tag_right(base)
    Hleft = hilbert_space(left_basis, args...; kwargs...)
    Hright = TransposedSpace(hilbert_space(right_basis, args...; kwargs...))
    Hfull = tensor_product(Hleft, Hright)
    return Hfull, Hleft, Hright, tag_left, tag_right
end
function open_system(H::AbstractHilbertSpace; kwargs...)
    Hleft = tag_left(H)
    Hright = TransposedSpace(tag_right(H))
    Hlr = tensor_product(Hleft, Hright; kwargs...)
    return Hlr, Hleft, Hright, tag_left, tag_right
end

"""
    dissipator(L)

The symbolic Lindblad dissipator `ρ ↦ LρL† - ½{L†L, ρ}` for the jump operator `L`, acting on the
vectorized density matrix of a space returned by [`open_system`](@ref). A prefactor belongs to `L`: `dissipator(c * L)` has rate `abs2(c)`. Public, not exported.

# Examples
```julia
using FermionicHilbertSpaces
using FermionicHilbertSpaces: open_system, dissipator
@fermions c
Hlr, Hl, Hr, left, right = open_system(c, 1:1)
representation(dissipator(c[1]), Hlr)
```
"""
dissipator(L) = tag_left(L) * tag_right(L') - (tag_left(L' * L) + tag_right(L' * L)) / 2

add_tag(expr, tag) = NonCommutativeProducts.ncmap(Base.Fix2(add_tag, tag), expr)


@testitem "TransposedSpace basics" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: TransposedSpace

    @fermions c
    H = hilbert_space(c, 1:1)
    Ht = TransposedSpace(H)
    op = c[1]' * c[1]
    @test representation(op, Ht) == transpose(representation(op, H))

    @boson b
    Hb = hilbert_space(b, 2)
    @test representation(b' * b, TransposedSpace(Hb)) == transpose(representation(b' * b, Hb))

    @spin s 1 // 2
    Hs = hilbert_space(s)
    @test representation(s[:x], TransposedSpace(Hs)) == transpose(representation(s[:x], Hs))
end

@testitem "add_tag and TransposedSpace of a product sector" begin
    using FermionicHilbertSpaces: add_tag, TransposedSpace
    @fermions f
    @bosons b
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(b[1], 3); constraint=NumberConservation(1))
    op = 2f[1]' * f[2] + f[2]' * b[1] + b[1]' * b[1]
    m = representation(op, H)
    @test representation(op, TransposedSpace(H)) == transpose(m)
    @test representation(add_tag(op, :t), add_tag(H, :t)) == m
end

@testitem "open_system symbolic left-right interface" begin
    using FermionicHilbertSpaces: open_system
    @fermions c
    Hfull, Hleft, Hright, left, right = open_system(c, 1:1)
    c_left = left(c)
    c_right = right(c)
    op = c[1]' * c[1]

    M = representation(left(op) * right(op) + left(op), Hfull)
    @test size(M) == (dim(Hfull), dim(Hfull))
    Mexpected = representation((c_left[1]' * c_left[1]) * (c_right[1]' * c_right[1]) + (c_left[1]' * c_left[1]), Hfull)
    @test M ≈ Mexpected
    @test reshape(reshape(M, Hfull => (Hleft, Hright)), (Hleft, Hright) => Hfull) == M

    @boson b
    Hb, Hleftb, Hrightb, left_b, right_b = open_system(b, 2)
    bl = left_b(b)
    br = right_b(b)
    op_b = b' * b
    Mb = representation(left_b(op_b) * right_b(op_b) + left_b(op_b), Hb)
    @test size(Mb) == (dim(Hb), dim(Hb))
    Mexpected_b = representation((bl' * bl) * (br' * br) + (bl' * bl), Hb)
    @test Mb ≈ Mexpected_b
    @test reshape(reshape(Mb, Hb => (Hleftb, Hrightb)), (Hleftb, Hrightb) => Hb) == Mb

    @spin s 1 // 2
    Hs, _, _ = open_system(s)
    s_left = left(s)
    s_right = right(s)
    H = tensor_product(Hfull, Hs)
    op = c[1]' * s[:z]
    M = representation(left(op) * right(op) + left(op), H)
    Mexpected = representation((c_left[1]' * s_left[:z]) * (c_right[1]' * s_right[:z]) + (c_left[1]' * s_left[:z]), H)
    @test M ≈ Mexpected
    @test reshape(reshape(M, H => (Hleft, Hs, Hright)), (Hleft, Hs, Hright) => H) == M
end

@testitem "open_system commutator of a single boson" begin
    using FermionicHilbertSpaces: open_system
    @boson a
    @bosons b
    Ha, _, _, left, right = open_system(a, 3)
    Hb, _, _, _, _ = open_system(b[1], 3)
    @test left(a) != right(a)
    @test representation(left(a' * a) - right(a' * a), Ha) == representation(left(b[1]' * b[1]) - right(b[1]' * b[1]), Hb)
end

@testitem "open_system on Majorana spaces" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: open_system

    @majoranas γ
    Hfull, Hleft, Hright, left, right = open_system(hilbert_space(γ, 1:4))
    @test dim(Hfull) == 16

    op = γ[1] * γ[2] + 0.5 * γ[3] * γ[4]
    M = representation(left(op) * right(op) + left(op), Hfull)
    @test size(M) == (16, 16)
    Mexpected = representation((left(γ[1]) * left(γ[2]) + 0.5 * left(γ[3]) * left(γ[4])) * (right(γ[1]) * right(γ[2]) + 0.5 * right(γ[3]) * right(γ[4])) + left(op), Hfull)
    @test M ≈ Mexpected
end

@testitem "Matrix representation of transposed spaces" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: TransposedSpace

    # An operator A*B acts as A*(B*v) on a normal space
    # On a transposed space, we change the order, so
    # A*B acts as B^T*A^T*v
    # Compare to the kron identity vec(X*A*B) = kron(B^T*A^T, I) * vec(X)
    @spin s 1 // 2
    H = hilbert_space(s)
    Ht = TransposedSpace(H)

    A = sum((n + 1im) * s[op] for (n, op) in zip(0:3, (:I, :x, :y, :z)))
    @test transpose(representation(A, H)) ≈ representation(A, Ht)

    B = sum((n + 2)^2 * s[op] for (n, op) in zip(0:3, (:I, :x, :y, :z)))
    @test representation(A * B, H) ≈ representation(A, H) * representation(B, H)
    @test representation(A * B, Ht) ≈ representation(B, Ht) * representation(A, Ht)
    @test representation(A * B, Ht) ≈ transpose(representation(A * B, H))
    # Test on a mixed space
    @fermions f
    Hf = hilbert_space(f, 1:2)
    Hfull = tensor_product(H, Hf)
    Hfull_t = TransposedSpace(Hfull)
    A2 = A * f[1]' * f[1]
    B2 = B * f[1]' * f[1] - I
    @test transpose(representation(A2, Hfull)) ≈ representation(A2, Hfull_t)
    @test representation(A2 * B2, Hfull_t) ≈ transpose(representation(A2 * B2, Hfull))

    @test representation(A2 * B2, Hfull) ≈ representation(A2, Hfull) * representation(B2, Hfull)
    @test representation(A2 * B2, Hfull_t) ≈ representation(B2, Hfull_t) * representation(A2, Hfull_t)

    # Test with constrained spaces
    _Hf = hilbert_space(f, 1:2)
    Hf = constrain_space(_Hf, collect(basisstates(_Hf))[[2, 3]])
    Hfull = tensor_product(H, Hf)
    Hfull_t = TransposedSpace(Hfull)
    A2 = A * f[1]' * f[1] + 0.5 * s[:x] * f[1]' * f[2]
    B2 = B * f[2]' * f[1] - I
    @test transpose(representation(A2, Hfull)) ≈ representation(A2, Hfull_t)
    @test representation(A2 * B2, Hfull_t) ≈ transpose(representation(A2 * B2, Hfull))
    @test representation(A2 * B2, Hfull) ≈ representation(A2, Hfull) * representation(B2, Hfull)
    @test representation(A2 * B2, Hfull_t) ≈ representation(B2, Hfull_t) * representation(A2, Hfull_t)

    # test with sector space
    Hf = hilbert_space(f, 1:2, NumberConservation(1))
    Hfull = tensor_product(H, Hf)
    Hfull_t = TransposedSpace(Hfull)
    A2 = A * f[1]' * f[1] + 0.5 * s[:x] * f[1]' * f[2]
    B2 = B * f[2]' * f[1] - I
    @test transpose(representation(A2, Hfull)) ≈ representation(A2, Hfull_t)
    @test representation(A2 * B2, Hfull_t) ≈ transpose(representation(A2 * B2, Hfull))
    @test representation(A2 * B2, Hfull) ≈ representation(A2, Hfull) * representation(B2, Hfull)
    @test representation(A2 * B2, Hfull_t) ≈ representation(B2, Hfull_t) * representation(A2, Hfull_t)

    # test with Majorana fermions
    @majoranas m
    Hm = hilbert_space(m, 1:4)
    Hm_t = TransposedSpace(Hm)
    A = m[1] * m[2] + 0.5 * m[3] * m[4] + (1 + 2im) * m[2] + 3im * m[1] * m[3] - m[2] * m[3] * m[4]
    @test transpose(representation(A, Hm)) ≈ representation(A, Hm_t)
end

@testitem "Vectorization and kron identity" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: open_system

    @spin s 1 // 2
    H = hilbert_space(s)
    Hfull, Hleft, Hright, left, right = open_system(s)

    A = 1.2 * s[:x] - 0.4im * s[:y] + 0.7 * s[:z]
    B = -0.3 * s[:x] + 1.1im * s[:y] + 0.2 * s[:z]

    MA = representation(A, H)
    MB = representation(B, H)

    Msuper = representation(left(A) * right(B), Hfull)
    @test Msuper ≈ kron(transpose(MB), MA)

    Mright = representation(right(A * B), Hfull)
    @test Mright ≈ kron(transpose(MA * MB), I(dim(H)))
end

@testitem "Lindblad steady state of a fermionic level" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: open_system, dissipator

    @fermions c
    γin, γout = 0.3, 0.7
    Hlr, Hl, Hr, left, right = open_system(c, 1:1)
    ham = 1.0 * c[1]'c[1]
    lindbladian = -1im * (left(ham) - right(ham)) + dissipator(√γin * c[1]') + dissipator(√γout * c[1])
    mat = representation(lindbladian, Hlr, Matrix)
    Ivec = reshape(I(dim(Hl)), (Hl, Hr) => Hlr)
    @test norm(mat' * Ivec) < 1e-12

    ρ = reshape(nullspace(mat)[:, 1], Hlr => (Hl, Hr))
    ρ ./= tr(ρ)
    @test ρ ≈ ρ'
    @test all(≥(-1e-12), eigvals(Hermitian(ρ)))
    @test tr(representation(c[1]'c[1], hilbert_space(c, 1:1)) * ρ) ≈ γin / (γin + γout)

    # Two modes with hopping and an odd jump operator, against the matrix formula
    # dρ/dt = -i[H,ρ] + LρL† - ½{L†L,ρ} on an even ρ
    Hlr, Hl, Hr, left, right = open_system(c, 1:2)
    ham = 0.4 * (c[1]'c[2] + c[2]'c[1]) + 0.3 * c[1]'c[1]
    L = 0.5 * c[1] + 0.8 * c[2]'
    mat = representation(-1im * (left(ham) - right(ham)) + dissipator(L), Hlr)
    H2 = hilbert_space(c, 1:2)
    Hm, Lm = representation(ham, H2, Matrix), representation(L, H2, Matrix)
    P = Diagonal([(-1)^count_ones(i - 1) for i in 1:dim(H2)]) # parity in the Fock basis
    X = rand(ComplexF64, 4, 4)
    ρ = (X + X' + P * (X + X') * P) / 4
    dρ = reshape(mat * reshape(ρ, (Hl, Hr) => Hlr), Hlr => (Hl, Hr))
    @test dρ ≈ -1im * (Hm * ρ - ρ * Hm) + Lm * ρ * Lm' - 0.5 * (Lm'Lm * ρ + ρ * Lm'Lm)
end

@testitem "open_system of a tensor product with a sector" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: open_system
    @fermions f
    @bosons a
    H = tensor_product(hilbert_space(f, 1:2, NumberConservation(1)), hilbert_space(a[1], 3))
    Hlr, Hl, Hr, left, right = open_system(H)
    dissipator(L) = left(L) * right(L') - 0.5 * (left(L'L) + right(L'L))
    ham = 0.4 * (f[1]'f[2] + f[2]'f[1]) + a[1]'a[1] + 0.3 * (f[1]'f[1] - f[2]'f[2]) * (a[1] + a[1]')
    L = 0.5 * a[1]
    mat = representation(-1im * (left(ham) - right(ham)) + dissipator(L), Hlr)
    Hm, Lm = representation(ham, H, Matrix), representation(L, H, Matrix)
    X = rand(ComplexF64, dim(H), dim(H))
    ρ = X + X'
    dρ = reshape(mat * reshape(ρ, (Hl, Hr) => Hlr), Hlr => (Hl, Hr))
    @test dρ ≈ -1im * (Hm * ρ - ρ * Hm) + Lm * ρ * Lm' - 0.5 * (Lm'Lm * ρ + ρ * Lm'Lm)
end

@testitem "Transposed constrained spaces" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: TransposedSpace, open_system
    @fermions c
    v = rand(4)
    for H in (hilbert_space(c, 1:3, NumberConservation(0:1)), hilbert_space(c, 1:3, ParityConservation(1)))
        for op in (c[1]' * c[2] + 0.5 * c[3]' * c[1], c[1]' * c[1] * c[2]' * c[3] + 2I)
            M = representation(op, H, Matrix)
            @test representation(op, TransposedSpace(H), Matrix) ≈ transpose(M)
            @test transpose(representation(op, H, :lazy)) * v ≈ transpose(M) * v
        end
    end
    H = hilbert_space(c, 1:3, NumberConservation(0:1))
    @test representation(c[1], TransposedSpace(H), Matrix) ≈ transpose(representation(c[1], H, Matrix))

    Hlr, Hl, Hr, left, right = open_system(H)
    X = rand(dim(H), dim(H))
    S = representation(left(c[2]) * right(c[1]), Hlr)
    @test reshape(S * reshape(X, (Hl, Hr) => Hlr), Hlr => (Hl, Hr)) ≈ representation(c[2], H, Matrix) * X * representation(c[1], H, Matrix)
    @test_throws ArgumentError representation(left(c[1]') * right(c[1]), Hlr)
end

function _apply_local_operators(op, state, space::TransposedSpace, precomp)
    newstate, amp = apply_local_operators(op, state, parent(space), precomp; transpose=true)
    return newstate, amp
end
# `newstate` with its parts on transposed groups replaced by those of `state`
_keep_transposed_parts(newstate, state, ::TransposedSpace) = state
_keep_transposed_parts(newstate, state, H::AbstractSubspace) = _keep_transposed_parts(newstate, state, parent(H))
_keep_transposed_parts(newstate::ProductState{B}, state::ProductState, H::ProductSpace) where {B} =
    ProductState{B}(map(_keep_transposed_parts, newstate.states, state.states, groups(H)))
_keep_transposed_parts(newstate, state, H) = newstate

rewrap(::TransposedSpace, space) = TransposedSpace(space)
@testitem "open_system of a custom algebra" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: @field, Op, AbstractBasisState, local_mul_effect,
        apply_local_operator, mat_eltype, add_tag, open_system, dissipator
    import FermionicHilbertSpaces.NonCommutativeProducts: Swap, AddTerms
    struct ShiftKind
        k::Int
    end
    struct NumKind end
    Base.adjoint(s::ShiftKind) = ShiftKind(-s.k)
    Base.adjoint(n::NumKind) = n
    local_mul_effect(a::Op{ShiftKind}, b::Op{ShiftKind}) = Op(ShiftKind(a.kind.k + b.kind.k), a.mode)
    local_mul_effect(a::Op{NumKind}, b::Op{ShiftKind}) = AddTerms((Swap(1), b.kind.k * b))
    struct Site <: AbstractBasisState
        n::Int
    end
    apply_local_operator(s::ShiftKind, state::Site, space, precomp) = (Site(mod(state.n + s.k, 3)), 1)
    apply_local_operator(::NumKind, state::Site, space, precomp) = (state, state.n)
    mat_eltype(::Type{ShiftKind}) = Int
    @field c::ShiftKind
    Shift(site, k) = c[site][k]
    Num(site) = Op(NumKind(), c[site])

    l(x) = add_tag(x, :l)
    r(x) = add_tag(x, :r)
    @test l(Num(1)) * l(Shift(1, 1)) == l(Shift(1, 1)) * l(Num(1)) + l(Shift(1, 1))
    @test l(Num(1)) * r(Shift(1, 1)) == r(Shift(1, 1)) * l(Num(1))
    @test l(r(Shift(1, 1))) != r(l(Shift(1, 1)))
    labels = (1, 2)
    H = hilbert_space(c, labels, Site.(0:2))
    Hlr, Hl, Hr, left, right = open_system(H)
    A = Shift(labels[1], 1) * Num(labels[2]) + 2 * Num(labels[1])
    B = Shift(labels[2], -1) + Shift(labels[1], 1) * Shift(labels[2], 1)
    MA, MB = representation(A, H, Matrix), representation(B, H, Matrix)
    ρ = randn(ComplexF64, dim(H), dim(H))
    v = reshape(ρ, (Hl, Hr) => Hlr)
    @test reshape(representation(left(A) * right(B), Hlr) * v, Hlr => (Hl, Hr)) ≈ MA * ρ * MB
    L = representation(-1im * (left(A + A') - right(A + A')) + dissipator(B), Hlr)
    @test abs(tr(reshape(L * v, Hlr => (Hl, Hr)))) < 1.0e-10
end

@testitem "open_system of a projector onto a state" begin
    using FermionicHilbertSpaces: open_system
    @fermions c
    H = hilbert_space(c, 1:2)
    ψ = Kets(H)("10") + 1im * Kets(H)("01")
    P = ψ * ψ'
    MP = representation(P, H, Matrix)
    Hlr, Hl, Hr, left, right = open_system(H)
    ρ = randn(ComplexF64, dim(H), dim(H))
    S = representation(left(P) * right(P'), Hlr)
    @test reshape(S * reshape(ρ, (Hl, Hr) => Hlr), Hlr => (Hl, Hr)) ≈ MP * ρ * MP'
end
