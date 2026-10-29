
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
TransposedSpace(H::AbstractSubspace) = rewrap(H, TransposedSpace(parent(H)))
state_mapper(H::TransposedSpace, Hs) = state_mapper(H.parent, Hs)
combine_states(states, H::TransposedSpace) = combine_states(states, H.parent)
Base.parent(H::TransposedSpace) = H.parent
partial_trace_phase_factor(f1, f2, H::TransposedSpace) = partial_trace_phase_factor(f1, f2, parent(H))
partial_trace_phase_factor_eltype(H::TransposedSpace) = partial_trace_phase_factor_eltype(parent(H))
partial_transpose_phase_factor_eltype(H::TransposedSpace) = partial_transpose_phase_factor_eltype(parent(H))
partial_transpose_phase_factor(f1, f2, H::TransposedSpace) = partial_transpose_phase_factor(f1, f2, parent(H))

function FermionicSpace(spaces::AbstractVector{F}, group) where {F<:TransposedSpace}
    only(unique(map(group_id, spaces))) == group || throw(ArgumentError("All spaces must belong to the group `$group`."))
    TransposedSpace(FermionicSpace(map(H -> H.parent, spaces), group))
end

combine_into_group(group::MajoranaGroup, spaces::AbstractVector{<:TransposedSpace}) =
    TransposedSpace(combine_into_group(group, map(parent, spaces)))


tag_left(base) = add_tag(base, :left)
tag_right(base) = add_tag(base, :right)

"""
    open_system(base, args...; kwargs...)
    open_system(H::AbstractHilbertSpace; kwargs...)

Return `(Hlr, Hleft, Hright, left, right)` for the vectorized density matrix of a system: `Hlr`
is the tensor product of a `:left` copy `Hleft` and a transposed `:right` copy `Hright` of the
space `hilbert_space(base, args...; kwargs...)`, or of `H`. `left(op)` and `right(op)` tag a symbolic
operator so it acts on `ρ` from the left or from the right, `representation(left(A) * right(B), Hlr)`
being the superoperator `ρ ↦ AρB`. Public, not exported.

# Examples
```julia
using FermionicHilbertSpaces: open_system
@fermions c
Hlr, Hl, Hr, left, right = open_system(c, 1:1)
ham = c[1]' * c[1]
representation(-1im * (left(ham) - right(ham)), Hlr) # the commutator ρ ↦ -i[ham, ρ]
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
vectorized density matrix of a space returned by [`open_system`](@ref). Public, not exported.

# Examples
```julia
using FermionicHilbertSpaces: open_system, dissipator
@fermions c
Hlr, Hl, Hr, left, right = open_system(c, 1:1)
representation(dissipator(c[1]), Hlr)
```
"""
dissipator(L) = tag_left(L) * tag_right(L') - (tag_left(L' * L) + tag_right(L' * L)) / 2

function add_tag(factor::AbstractSym, tag)
    basis = symbolic_basis(factor)
    new_basis = add_tag(basis, tag)
    return with_basis(factor, new_basis)
end
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

    A = sum((n + 1im) * s[n] for n in 0:3)
    @test transpose(representation(A, H)) ≈ representation(A, Ht)

    B = sum((n + 2)^2 * s[n] for n in 0:3)
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
    mat = representation(lindbladian, Hlr, :dense)
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
    Hm, Lm = representation(ham, H2, :dense), representation(L, H2, :dense)
    P = Diagonal([(-1)^count_ones(i - 1) for i in 1:dim(H2)]) # parity in the Fock basis
    X = rand(ComplexF64, 4, 4)
    ρ = (X + X' + P * (X + X') * P) / 4
    dρ = reshape(mat * reshape(ρ, (Hl, Hr) => Hlr), Hlr => (Hl, Hr))
    @test dρ ≈ -1im * (Hm * ρ - ρ * Hm) + Lm * ρ * Lm' - 0.5 * (Lm'Lm * ρ + ρ * Lm'Lm)
end

function _apply_local_operators(op, state, space::TransposedSpace, precomp)
    newstate, amp = apply_local_operators(op, state, parent(space), precomp; transpose=true)
    return newstate, amp
end

maximum_particles(space::TransposedSpace) = maximum_particles(parent(space))
rewrap(::TransposedSpace, space) = TransposedSpace(space)