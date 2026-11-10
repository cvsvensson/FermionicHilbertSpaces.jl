"""
    Field{O}(name, group=nothing, params=nothing)

A named family of modes for operators of type `O`, created by [`@field`](@ref) or by macros such as
[`@bosons`](@ref). `field[l]` is the mode labelled `l`, or the operator an algebra puts there, such as the
annihilation operator `b[l]`. Fields of one `group` (e.g. `@fermions a b`) share one space, which their algebra
builds with `combine_into_group(::Group{O}, atoms)`, and `params` holds parameters of the algebra such as the spin `J`.
"""
struct Field{A, G, P, T <: Tags}
    name::Symbol
    group::G
    params::P
    tags::T
end
Field{O}(name::Symbol, group = nothing, params = nothing, tags::Tags = Tags(nothing)) where {O} =
    Field{O, typeof(group), typeof(params), typeof(tags)}(name, group, params, tags)

"""
    Mode(field, label)

The mode of `field` labelled `label`. A mode without a field, such as `@boson b`, has the label `()`.
For a field of operators of type `O`, `field[l][k]` is the operator `O(field[l], k)`.
"""
struct Mode{F <: Field, L}
    field::F
    label::L
end

"""
    @field c::O

Create the field `c` of operators of type `O`: `c[l]` is the mode labelled `l`, `c[l][k]` the operator
`O(c[l], k)`, and `hilbert_space(c, labels, states)` the product of one space with basis `states` per mode.
"""
macro field(ex)
    Meta.isexpr(ex, :(::), 2) || throw(ArgumentError("Use `@field c::O`, got `$ex`."))
    name, O = ex.args
    return :($(esc(name)) = Field{$(esc(O))}($(QuoteNode(name))))
end

"""
    mode(op)

The [`Mode`](@ref) that the operator `op` acts on, which gives its `symbolic_group`, ordering and
printing. `with_mode(op, mode)`, the operator on another mode, is needed for tags and [`open_system`](@ref).
"""
function mode end
mode(m::Mode) = m
"""
    with_mode(op, mode)

The operator `op` acting on `mode` instead. See [`mode`](@ref).
"""
function with_mode end

# What `field[l]` returns for the mode `m` of label `l`.
_field_entry(m::Mode) = m
Base.getindex(f::Field, l) = _field_entry(Mode(f, l))
Base.getindex(f::Field, ls...) = _field_entry(Mode(f, ls))
Base.getindex(::Field, l::AbstractSym) = throw(ArgumentError("Modes are indexed by labels, not by operators, got `$l`."))
function Base.getindex(m::Mode{<:Field{O}}, k) where {O}
    _field_entry(m) === m || throw(ArgumentError("`$(m.field)[l]` is already an operator, so it cannot be indexed further."))
    return O(m, k)
end
label(m::Mode) = m.label
tags(f::Field) = f.tags
tags(m::Mode) = tags(m.field)

Base.:(==)(::Field, ::Field) = false
Base.:(==)(a::Field{O}, b::Field{O}) where {O} = a.name == b.name && a.group == b.group && a.params == b.params && a.tags == b.tags
Base.hash(f::Field{O}, h::UInt) where {O} = hash(f.tags, hash(f.params, hash(f.group, hash(f.name, hash(O, h)))))
Base.:(==)(a::Mode, b::Mode) = a.field == b.field && a.label == b.label
Base.hash(m::Mode, h::UInt) = hash(m.label, hash(m.field, h))
_order_key(f::Field{O}) where {O} = (nameof(O), f.name, tagset(f), f.params, f.group)
Base.isless(a::Mode, b::Mode) = _order_isless((_order_key(a.field), a.label), (_order_key(b.field), b.label))

add_tag(f::Field{O}, tag) where {O} = Field{O}(f.name, f.group, f.params, add_tag(f.tags, tag))
add_tag(m::Mode, tag) = Mode(add_tag(m.field, tag), m.label)
tag_label(m::Mode, tag) = add_tag(m, tag)

Base.show(io::IO, f::Field) = print(io, _symbolic_name_with_tags(f.name, f))
Base.show(io::IO, m::Mode) = (show(io, m.field); _show_label(io, m.label))
_show_label(io, ::Tuple{}) = nothing
_show_label(io, l::Tuple) = Base.show_delim_array(io, l, "[", ",", "]", false)
_show_label(io, l) = print(io, "[", l, "]")

"""
    Group{O}(names, tags)

The group key of fields that share one space, such as one `@fermions a b` block: it equals `group_id` of that
space, and `combine_into_group(::Group{O}, atoms)` builds it. A mode of an ungrouped field is its own group.
"""
struct Group{O, G, T <: Tags}
    names::G
    tags::T
end
Group{O}(names, tags::Tags) where {O} = Group{O, typeof(names), typeof(tags)}(names, tags)
Group(f::Field{O}) where {O} = Group{O}(f.group, f.tags)
Base.:(==)(::Group, ::Group) = false
Base.:(==)(a::Group{O}, b::Group{O}) where {O} = a.names == b.names && a.tags == b.tags
Base.hash(g::Group{O}, h::UInt) where {O} = hash(g.tags, hash(g.names, hash(O, h)))
Base.isless(a::Group{O}, b::Group{P}) where {O, P} = _order_isless((nameof(O), a.names, tagset(a)...), (nameof(P), b.names, tagset(b)...))
tags(g::Group) = g.tags
add_tag(g::Group{O}, tag) where {O} = Group{O}(g.names, add_tag(g.tags, tag))

symbolic_group(m::Mode) = isnothing(m.field.group) ? m : Group(m.field)
symbolic_group(op::AbstractSym) = symbolic_group(mode(op))
atomic_id(m::Mode) = m
_atom_names(m::Mode) = (m,)
add_tag(op::AbstractSym, tag) = with_mode(op, add_tag(mode(op), tag))

# The methods of `hilbert_space(field, labels, ...)` take a vector of labels. Other collections, such as tuples,
# generators and matrices, are flattened into one. The order of the labels is the order of the basis, so
# unordered collections are rejected.
function hilbert_space(f::Field, labels, args...; kwargs...)
    labels isa AbstractVector && throw(MethodError(hilbert_space, (f, labels, args...)))
    (labels isa Union{AbstractString, AbstractChar, Number, AbstractSet, AbstractDict} || !applicable(iterate, labels)) &&
        throw(ArgumentError("`labels` must be an ordered collection of labels such as `1:3` or `[:a, :b]`, got `$(repr(labels))`."))
    labels = vec(collect(labels))
    # An empty tuple collects to a `Vector{Union{}}`, which matches the methods for every element type.
    return hilbert_space(f, labels isa Vector{Union{}} ? Any[] : labels, args...; kwargs...)
end
# The product of the spaces `hilbert_space(f[l], args...)` of the modes of `f`, constrained by `constraint`.
_mode_product(f::Field, labels, constraint, args...) =
    tensor_product(map(l -> hilbert_space(Mode(f, l), args...), labels); constraint)
hilbert_space(f::Field, labels::AbstractVector, x, @nospecialize(constraint::ConstraintLike = NoSymmetry())) = _mode_product(f, labels, constraint, x)
function hilbert_space(m::Mode, states)
    (_field_entry(m) === m && states isa Union{AbstractVector, Tuple, BasisStates}) || throw(MethodError(hilbert_space, (m, states)))
    isnothing(m.field.group) || throw(ArgumentError("The grouped field `$(m.field)` needs its own space type and a `combine_into_group(::Group{O}, atoms)` method."))
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
    @test_throws MethodError hilbert_space(f, 1:3, 1)
    @test_throws MethodError hilbert_space(b, 1:2, 3.0)
    @test_throws MethodError hilbert_space(b[1], [0, 1])
    for labels in (:a, "ab", 1, Set(1:2), Dict(1 => 2))
        @test_throws ArgumentError hilbert_space(f, labels)
    end
    @test hilbert_space(f, ()) == hilbert_space(f, Int[])
end

@testitem "Custom operators on a field" begin
    import FermionicHilbertSpaces: @field, AbstractSym, AbstractBasisState, apply_local_operator, mode, with_mode,
        add_tag, open_system
    import FermionicHilbertSpaces.NonCommutativeProducts: @nc
    struct Shift{M} <: AbstractSym
        mode::M
        k::Int
    end
    @nc Shift
    mode(s::Shift) = s.mode
    with_mode(s::Shift, m) = Shift(m, s.k)
    Base.adjoint(s::Shift) = Shift(s.mode, -s.k)
    struct Site <: AbstractBasisState
        n::Int
    end
    apply_local_operator(s::Shift, state::Site, space, precomp) = (Site(mod(state.n + s.k, 3)), 1)

    @field c::Shift
    @test c[1][2] == Shift(c[1], 2)
    @test c[2][1] * c[1][1] == c[1][1] * c[2][1]
    @test repr(mode(add_tag(c[1][1], :l))) == "c(l)[1]"
    H = hilbert_space(c, 1:2, Site.(0:2))
    @test H == tensor_product([GenericHilbertSpace(c[i], Site.(0:2)) for i in 1:2])
    @test_throws ArgumentError hilbert_space(FermionicHilbertSpaces.Field{Shift}(:g, (:g,)), 1:2, Site.(0:2))
    A = c[1][1] * c[2][-1]
    MA = representation(A, H, Matrix)
    Hlr, Hl, Hr, left, right = open_system(H)
    ρ = randn(dim(H), dim(H))
    v = reshape(ρ, (Hl, Hr) => Hlr)
    @test reshape(representation(left(A) * right(A'), Hlr) * v, Hlr => (Hl, Hr)) ≈ MA * ρ * MA'
end
