"""
    Field{K}(name, group=nothing, params=nothing, tags=Tags(nothing))

A named family of modes for operators of kind `K`, created by [`@field`](@ref) or by macros such as
[`@bosons`](@ref). `field[l]` is the mode labelled `l`, or the operator of that mode if `K` has a
[`mode_operator`](@ref), such as the annihilation operator `b[l]`. Fields of one `group` (e.g. `@fermions a b`)
share one space, which their algebra builds with `combine_into_group(::Group{K}, atoms)`, and `params` holds
parameters of the algebra such as the spin `J`.
"""
struct Field{K, G, P, T <: Tags}
    name::Symbol
    group::G
    params::P
    tags::T
end
Field{K}(name::Symbol, group = nothing, params = nothing, tags::Tags = Tags(nothing)) where {K} =
    Field{K, typeof(group), typeof(params), typeof(tags)}(name, group, params, tags)

"""
    Mode(field, label)

The mode of `field` labelled `label`. A mode without a field, such as `@boson b`, has the label `()`.
For a field of operators of kind `K`, `field[l][k]` is the operator `Op(K(k), field[l])`.
"""
struct Mode{F <: Field, L}
    field::F
    label::L
end

"""
    Op(kind, mode)

The operator that acts as `kind` on `mode`. The kind says what the operator is, such as `Fermion(true)` for a
creation operator, and defines `adjoint`, [`apply_local_operator`](@ref) and [`mat_eltype`](@ref).
"""
struct Op{K, M} <: AbstractSym
    kind::K
    mode::M
end
# GPU kernels receive the kind in place of the operator, so code that runs on both reads `kind(x)`.
kind(op::Op) = op.kind
kind(k) = k

"""
    @field c::K

Create the field `c` of operators of kind `K`: `c[l]` is the mode labelled `l`, `c[l][k]` the operator
`Op(K(k), c[l])`, and `hilbert_space(c, labels, states)` the product of one space with basis `states` per mode.
"""
macro field(ex)
    Meta.isexpr(ex, :(::), 2) || throw(ArgumentError("Use `@field c::K`, got `$ex`."))
    name, K = ex.args
    return :($(esc(name)) = Field{$(esc(K))}($(QuoteNode(name))))
end

"""
    mode_operator(::Type{K})

The kind of the operator that `field[l]` returns when a mode of kind `K` has a single operator, as in
`mode_operator(::Type{Boson}) = Boson(-1)` for the annihilation operator `b[l]`. The default `nothing` makes
`field[l]` a [`Mode`](@ref) with the operators `field[l][k]`.
"""
mode_operator(::Type) = nothing

"""
    mode(op)

The [`Mode`](@ref) that the operator `op` acts on.
"""
mode(m::Mode) = m
mode(op::Op) = op.mode

_field_entry(m::Mode{<:Field{K}}) where {K} = (k = mode_operator(K); isnothing(k) ? m : Op(k, m))
Base.getindex(f::Field, l) = _field_entry(Mode(f, l))
Base.getindex(f::Field, ls...) = _field_entry(Mode(f, ls))
Base.getindex(::Field, l::AbstractSym) = throw(ArgumentError("Modes are indexed by labels, not by operators, got `$l`."))
function Base.getindex(m::Mode{<:Field{K}}, k) where {K}
    isnothing(mode_operator(K)) || throw(ArgumentError("`$(m.field)[l]` is already an operator, so it cannot be indexed further."))
    return Op(K(k), m)
end
Base.getindex(op::Op, k) = throw(ArgumentError("`$op` is already an operator, so it cannot be indexed further."))
label(m::Mode) = m.label
label(op::Op) = label(op.mode)
tags(f::Field) = f.tags
tags(m::Mode) = tags(m.field)

Base.:(==)(::Field, ::Field) = false
Base.:(==)(a::Field{K}, b::Field{K}) where {K} = a.name == b.name && a.group == b.group && a.params == b.params && a.tags == b.tags
# Few fields, since this runs in every Dict lookup of a symbolic term; `==` compares the others.
Base.hash(f::Field, h::UInt) = hash(f.name, hash(f.tags, h))
Base.:(==)(a::Mode, b::Mode) = a.field == b.field && a.label == b.label
Base.hash(m::Mode, h::UInt) = hash(m.label, hash(m.field, h))
Base.:(==)(a::Op, b::Op) = a.kind == b.kind && a.mode == b.mode
Base.hash(op::Op, h::UInt) = hash(op.kind, hash(op.mode, h))
_order_key(f::Field{K}) where {K} = (nameof(K), f.name, tagset(f), f.params, f.group)
Base.isless(a::Mode, b::Mode) = _order_isless((_order_key(a.field), a.label), (_order_key(b.field), b.label))

add_tag(f::Field{K}, tag) where {K} = Field{K}(f.name, f.group, f.params, add_tag(f.tags, tag))
add_tag(m::Mode, tag) = Mode(add_tag(m.field, tag), m.label)
add_tag(op::Op, tag) = Op(op.kind, add_tag(op.mode, tag))
tag_label(m::Mode, tag) = add_tag(m, tag)

Base.show(io::IO, f::Field) = print(io, _symbolic_name_with_tags(f.name, f))
Base.show(io::IO, m::Mode) = (show(io, m.field); _show_label(io, m.label))
_show_label(io, ::Tuple{}) = nothing
_show_label(io, l::Tuple) = Base.show_delim_array(io, l, "[", ",", "]", false)
_show_label(io, l) = print(io, "[", l, "]")
Base.show(io::IO, op::Op) = print(io, op.mode, "[", op.kind, "]")

Base.adjoint(op::Op) = Op(adjoint(op.kind), op.mode)
mat_eltype(::Type{<:Op{K}}) where {K} = mat_eltype(K)
apply_local_operator(op::Op, state, space, precomp) = apply_local_operator(op.kind, state, space, precomp)
apply_local_operator(op::Op, state, space, precomp, ::Type{P}) where {P} = apply_local_operator(op.kind, state, space, precomp, P)
# Reached when a kind has no method for the state. A method on `Op{K}` is not called, since `Op` forwards to its kind.
function apply_local_operator(kind::K, state, space, precomp) where {K}
    kind isa AbstractSym && throw(MethodError(apply_local_operator, (kind, state, space, precomp)))
    throw(ArgumentError("No `apply_local_operator` method for the kind `$K` on states of type `$(typeof(state))`. Define `apply_local_operator(kind::$K, state, space, precomp)` on the kind, not on `Op{$K}`."))
end

"""
    Group{K}(names, tags)

The group key of fields that share one space, such as one `@fermions a b` block: it equals `group_id` of that
space, and `combine_into_group(::Group{K}, atoms)` builds it. A mode of an ungrouped field is its own group.
"""
struct Group{K, G, T <: Tags}
    names::G
    tags::T
end
Group{K}(names, tags::Tags) where {K} = Group{K, typeof(names), typeof(tags)}(names, tags)
Group(f::Field{K}) where {K} = Group{K}(f.group, f.tags)
Base.:(==)(::Group, ::Group) = false
Base.:(==)(a::Group{K}, b::Group{K}) where {K} = a.names == b.names && a.tags == b.tags
Base.hash(g::Group{K}, h::UInt) where {K} = hash(g.tags, hash(g.names, hash(K, h)))
Base.isless(a::Group{K}, b::Group{P}) where {K, P} = _order_isless((nameof(K), a.names, tagset(a)...), (nameof(P), b.names, tagset(b)...))
tags(g::Group) = g.tags
add_tag(g::Group{K}, tag) where {K} = Group{K}(g.names, add_tag(g.tags, tag))

symbolic_group(m::Mode) = isnothing(m.field.group) ? m : Group(m.field)
symbolic_group(op::Op) = symbolic_group(op.mode)
atomic_id(x::Union{Mode, Op}) = mode(x)
_atom_names(x::Union{Mode, Op}) = (mode(x),)

# The methods of `hilbert_space(field, labels, ...)` take a vector of labels. Other collections, such as tuples,
# generators and matrices, are flattened into one. The order of the labels is the order of the basis, so
# unordered collections are rejected.
function hilbert_space(f::Field, labels, args...; kwargs...)
    isempty(kwargs) || throw(ArgumentError("`hilbert_space` takes no keyword arguments, got `$(join(keys(kwargs), ", "))`. A constraint is the last positional argument."))
    labels isa AbstractVector && _no_space(f, labels, args...)
    (labels isa Union{AbstractString, AbstractChar, Number, AbstractSet, AbstractDict} || !applicable(iterate, labels)) &&
        throw(ArgumentError("`labels` must be an ordered collection of labels such as `1:3` or `[:a, :b]`, got `$(repr(labels))`."))
    labels = vec(collect(labels))
    # An empty tuple collects to a `Vector{Union{}}`, which matches the methods for every element type.
    return hilbert_space(f, labels isa Vector{Union{}} ? Any[] : labels, args...)
end
# The product of the spaces `hilbert_space(f[l], args...)` of the modes of `f`, constrained by `constraint`.
_mode_product(f::Field, labels, constraint, args...) =
    tensor_product(map(l -> hilbert_space(Mode(f, l), args...), labels); constraint)
hilbert_space(f::Field, labels::AbstractVector, x, @nospecialize(constraint::ConstraintLike = NoSymmetry())) = _mode_product(f, labels, constraint, x)
hilbert_space(f::Field) = throw(ArgumentError("The field `$f` needs labels. See `?hilbert_space`."))
hilbert_space(op::Op, args...) = hilbert_space(op.mode, args...)
hilbert_space(m::Mode, args...) = _no_space(m, args...)
# The error for arguments that no method of `hilbert_space` accepts.
function _no_space(x, args...)
    got = isempty(args) ? "without arguments" : "from `" * join((repr(a; context = :limit => true) for a in args), ", ") * "`"
    throw(ArgumentError("`hilbert_space` cannot build a space for `$x` $got. See `?hilbert_space`."))
end
function hilbert_space(m::Mode{<:Field{K}}, states) where {K}
    states isa Union{AbstractVector, Tuple, BasisStates} || _no_space(m, states)
    isnothing(m.field.group) || throw(ArgumentError("The grouped field `$(m.field)` needs its own space type and a `combine_into_group(::Group{K}, atoms)` method."))
    return GenericHilbertSpace(m, states)
end

@testitem "Fields and modes" begin
    using FermionicHilbertSpaces: add_tag
    @spins s 1 // 2
    @spin S
    @bosons b
    @test add_tag(s, :l) != s && add_tag(s, :l)[1] == add_tag(s[1], :l)
    @test repr(add_tag(S[:z], :l)) == "S(l)[:z]"
    @test repr(b[1, :a]') == "b†[1, :a]"
    @test_throws ArgumentError FermionicHilbertSpaces.Mode(b, 1)[2]
    @test hilbert_space(b, (1, 2), 2) == hilbert_space(b, [1, 2], 2) == hilbert_space(b, (i for i in 1:2), 2)
    @fermions f
    @test_throws ArgumentError hilbert_space(b, 1:2)
    @test_throws ArgumentError hilbert_space(b[1])
    @test_throws ArgumentError hilbert_space(f[1], 2)
    @test_throws ArgumentError hilbert_space(f)
    @test_throws "keyword" hilbert_space(f, 1:3; constraint = NumberConservation(1))
    for labels in (:a, "ab", 1, Set(1:2), Dict(1 => 2))
        @test_throws ArgumentError hilbert_space(f, labels)
    end
    @test hilbert_space(f, ()) == hilbert_space(f, Int[])
    @test repr(hilbert_space((@fermion h)); context = :compact => true) == "h"
    @fermions g
    @test_throws ArgumentError FermionicHilbertSpaces.FermionicSpace(f, [g[1]])
end

@testitem "Custom operators on a field" begin
    import FermionicHilbertSpaces: @field, Op, AbstractBasisState, apply_local_operator, local_mul_effect, mode,
        add_tag, open_system
    import FermionicHilbertSpaces.NonCommutativeProducts: Swap
    struct Shift
        k::Int
    end
    Base.adjoint(s::Shift) = Shift(-s.k)
    struct Site <: AbstractBasisState
        n::Int
    end
    apply_local_operator(s::Shift, state::Site, space, precomp) = (Site(mod(state.n + s.k, 3)), 1)
    local_mul_effect(a::Shift, b::Shift) = a.k == -b.k ? 1 : a.k > b.k ? Swap(1) : nothing

    @field c::Shift
    @test c[1][2] == Op(Shift(2), c[1])
    @test c[2][1] * c[1][1] == c[1][1] * c[2][1]
    struct NoAction end
    @test_throws "on the kind, not on `Op{" representation(Op(NoAction(), c[1]), hilbert_space(c, 1:1, Site.(0:2)))
    @test repr(mode(add_tag(c[1][1], :l))) == "c(l)[1]"
    H = hilbert_space(c, 1:2, Site.(0:2))
    @test H == tensor_product([GenericHilbertSpace(c[i], Site.(0:2)) for i in 1:2])
    @test_throws ArgumentError hilbert_space(FermionicHilbertSpaces.Field{Shift}(:g, (:g,)), 1:2, Site.(0:2))
    struct Lower end
    FermionicHilbertSpaces.mode_operator(::Type{Lower}) = Lower()
    @field a::Lower
    @test hilbert_space(a, 1:2, Site.(0:2)) == tensor_product([hilbert_space(a[i], Site.(0:2)) for i in 1:2])
    A = c[1][1] * c[2][-1]
    MA = representation(A, H, Matrix)
    @test c[1][2] * c[1][1] == c[1][1] * c[1][2] && c[1][1] * c[1][-1] == 1
    @test representation(c[1][2] * c[1][1], H, Matrix) == representation(c[1][2], H, Matrix) * representation(c[1][1], H, Matrix)
    Hlr, Hl, Hr, left, right = open_system(H)
    ρ = randn(dim(H), dim(H))
    v = reshape(ρ, (Hl, Hr) => Hlr)
    @test reshape(representation(left(A) * right(A'), Hlr) * v, Hlr => (Hl, Hr)) ≈ MA * ρ * MA'
end
