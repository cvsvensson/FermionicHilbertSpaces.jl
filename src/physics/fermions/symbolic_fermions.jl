
"""
    @fermions a b ...

Create one or more fermion species with the given names. Indexing into fermions species
gives a concrete fermion. Fermions in one `@fermions` block anticommute with each other, 
and commute with fermions in other `@fermions` blocks.
`a[l]` is the annihilation operator of the mode labelled `l`, `a[l]'` its creation operator, and `a[i, σ]` is the mode labelled `(i, σ)`.

# Examples:
- `@fermions a b` creates two species of fermions that anticommute:
    - `a[1]' * a[1] + a[1] * a[1]' == 1`
    - `a[1]' * b[1] + b[1] * a[1]' == 0`
- `@fermions a; @fermions b` creates two species of fermions that commute with each other:
    - `a[1]' * a[1] + a[1] * a[1]' == 1`
    - `a[1] * b[1] - b[1] * a[1] == 0`

See also [`@majoranas`](@ref).
"""
macro fermions(xs...)
    defs = map(xs) do x
        :($(esc(x)) = Field{Fermion}($(Expr(:quote, x)), $xs))
    end
    fermions = Expr(:block, defs...,
        :(tuple($(map(x -> esc(x), xs)...))))
    length(defs) == 1 && return only(defs)
    return fermions
end

"""
    @fermion f

Create the annihilation operator `f` of a single fermionic mode, with space `hilbert_space(f)`. It is the
mode `f[]` of `@fermions f`.
"""
macro fermion(x)
    return :($(esc(x)) = Field{Fermion}($(QuoteNode(x)), ($(QuoteNode(x)),))[])
end

struct Fermion
    creation::Bool
end
const FermionSym{M} = Op{Fermion, M}
mode_operator(::Type{Fermion}) = Fermion(false)
Base.adjoint(k::Fermion) = Fermion(!k.creation)

Base.show(io::IO, x::FermionSym) = (show(io, x.mode.field); print(io, x.kind.creation ? "†" : ""); _show_label(io, x.mode.label))
function Base.isless(a::FermionSym, b::FermionSym)
    ga, gb = symbolic_group(a), symbolic_group(b)
    return if ga != gb
        ga < gb
    elseif a.kind.creation == b.kind.creation
        fa, fb = a.mode.field, b.mode.field
        fa.name == fb.name && return _order_isless(a.mode.label, b.mode.label)
        fa.name < fb.name
    else
        a.kind.creation > b.kind.creation
    end
end

hilbert_space(f::FermionSym) = hilbert_space(f.mode)
hilbert_space(m::Mode{<:Field{Fermion}}) = FermionicSpace([m], symbolic_group(m))

# The sign λ in a * b = λ * b * a + scalar
exchange_phase(a::FermionSym, b::FermionSym) = symbolic_group(a) == symbolic_group(b) ? -1 : 1
function NonCommutativeProducts.mul_effect(a::FermionSym, b::FermionSym)
    if a == b
        0
    elseif a < b
        nothing
    elseif a > b
        swap = Swap(exchange_phase(a, b))
        if a.mode == b.mode
            return AddTerms((swap, 1))
        else
            return swap
        end
    else
        return nothing
    end
end

mat_eltype(::Type{Fermion}) = Int

@testitem "SymbolicFermions" begin
    using Symbolics, LinearAlgebra
    @fermions f c
    @fermions b
    @test sprint(show, f) == "f"
    @variables a::Real z::Complex
    f1 = f[:a]
    f2 = f[:b]
    f3 = f[1, :↑]

    @test 1 * f1 == f1
    @test 1 * f1 + 0 == f1
    @test 1 * f1 + 0 == 1 * f1
    @test hash(f1) == hash(1 * f1) == hash(1 * f1 + 0)

    # Test canonical commutation relations
    @test f1' * f1 + f1 * f1' == 1
    @test iszero(f1 * f2 + f2 * f1)
    @test iszero(f1' * f2 + f2 * f1')

    # c anticommutes with f
    @test iszero(f1' * c[1] + c[1] * f1')
    # b commutes with f
    @test iszero(f1' * b[1] - b[1] * f1')

    @test_nowarn display(f1)
    @test_nowarn display(f3)
    @test_nowarn display(1 * f1)
    @test_nowarn display(2 * f3)
    @test_nowarn display(1 + f1)
    @test_nowarn display(1 + f3)
    @test_nowarn display(1 + a * f2 - 5 * f1 + 2 * z * f1 * f2)
    @test iszero(f1 - f1)
    @test iszero(f1 * f1)
    @test iszero(2 * f1 - 2 * f1)
    @test iszero(0 * f1)
    @test iszero(f1 * 0)
    @test iszero(f1^2)
    @test iszero(0 * (f1 + f2))
    @test iszero((f1 + f2) * 0)
    @test iszero(f1 * f2 * f1)
    f12 = f1 * f2
    @test iszero(f12'' - f12)
    @test iszero(f12 * f12)
    @test iszero(f12' * f12')
    nf1 = f1' * f1
    @test nf1^2 == nf1

    @test 1 + (f1 + f2) == 1 + f1 + f2 == f1 + f2 + 1 == f1 + 1 + f2 == 1 * f1 + f2 + 1 == f1 + 0.5 * f2 + 1 + (0 * f1 + 0.5 * f2) == (0.5 + 0.5 * f1 + 0.2 * f2) + 0.5 + (0.5 * f1 + 0.8 * f2) == (1 + f1' + (1 * f2)')'
    @test iszero((2 * f1) * (2 * f1))
    @test iszero((2 * f1)^2)
    @test (2 * f2) * (2 * f1) == -4 * f1 * f2
    @test f1 == (f1 * (f1 + 1)) == (f1 + 1) * f1
    @test iszero(f1 * (f1 + f2) * f1)
    @test (f1 * (f1 + f2)) == f1 * f2
    @test (2nf1 - 1) * (2nf1 - 1) == 1

    @test (1 * f1) * f2 == f1 * f2
    @test (1 * f1) * (1 * f2) == f1 * f2
    @test f1 * f2 == f1 * (1 * f2) == f1 * f2
    @test f1 - 1 == (1 * f1) - 1 == (0.5 + f1) - 1.5

    for op in [f1, f1 + f2, f1' * f2, f1 * f2 + 1]
        @test op + 1.0I == op + 1.0
        @test op - 1.0I == op - 1.0 == -(1.0I - op) == -(1.0 - op)
    end

end

@testitem "Single fermionic mode" begin
    @fermion f
    @fermions g
    @test representation(f' * f, hilbert_space(f), Matrix) == [0 0; 0 1]
    @test f * f' + f' * f == 1
    @test f * g[1] == g[1] * f
end

@testitem "Deterministic ordering of groups" begin
    using FermionicHilbertSpaces: add_tag, symbolic_group, SymbolicState
    @fermions a
    @fermions b
    @fermions c d
    @majoranas γ

    a_old = a
    @fermions a
    @test a == a_old && a[1] == a_old[1] && hash(a[1]) == hash(a_old[1])
    @test c[1] * d[1] == -d[1] * c[1]

    # groups are in lexicographic order of the names given to the macro
    g(x) = symbolic_group(x[1])
    @test g(a) < g(b) < g(c) == g(d)
    @test a[2] < b[1] && b[1] < c[1] && !(b[1] < a[2])
    @test a[1]' < b[1]
    @test only(keys((b[1] * a[1]).dict)).factors == [a[1], b[1]]
    @test only(keys((c[1] * b[1]' * a[2]).dict)).factors == [a[2], b[1]', c[1]]

    # the order is total and consistent with ==
    ts = [
        g(a), g(b), g(c), g(γ), g(add_tag(a, :left)), g(add_tag(a, :right)), g(add_tag(add_tag(a, :left), :right)),
        g(add_tag(γ, :left)),
    ]
    for x in ts, y in ts
        @test (x == y) == (!(x < y) && !(y < x))
        @test !(x < y && y < x)
    end
    @test issorted(sort(ts))
    @test sort(ts) == sort(reverse(ts))

    # symbolic states are ordered by group, space, ket and bra
    @boson bb
    Hf = hilbert_space(a, 1:1)
    Hb = hilbert_space(bb, 3)
    vf, vb = Kets(Hf), Kets(Hb)
    @test vf("0") < vf("1")
    @test vf("0")' < vf("0")
    states = [vf("0"), vf("1"), vf("0")', vf("1")', SymbolicState(Hf, vf("1").ket, vf("0").ket),
        vb("0"), vb("2"), vb("1")']
    for x in states, y in states
        @test (x == y) == (!(x < y) && !(y < x))
    end
    @test issorted(sort(states))
end

@testitem "Fermionic triple products vs matrices" begin
    @fermions f c
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(c, 1:1))
    ops = [f[1], f[1]', f[2], f[2]', c[1], c[1]']
    mats = [representation(op, H) for op in ops]
    for i in eachindex(ops), j in eachindex(ops), k in eachindex(ops)
        @test representation(ops[i] * ops[j] * ops[k], H) ≈ mats[i] * mats[j] * mats[k]
    end
end

@testitem "Products with mixed labels and tags" begin
    import FermionicHilbertSpaces: add_tag
    @fermions f
    @majoranas γ
    @bosons b
    @spins s 1 // 2
    @test iszero(f[1] * f[:a] + f[:a] * f[1])
    @test iszero(f[1] * f[1, :a] + f[1, :a] * f[1])
    @test iszero(γ[1] * γ[:a] + γ[:a] * γ[1])
    @test b[1] * b[:a] == b[:a] * b[1]
    @test s[1][:z] * s[:a][:z] == s[:a][:z] * s[1][:z]
    H = hilbert_space(f, [1, :a])
    @test representation(f[1] * f[:a], H) == representation(f[1], H) * representation(f[:a], H)

    # Tags of different types: operators on different groups commute
    tagged(x, y) = x * y == y * x
    @test tagged(add_tag(f, 1)[1], add_tag(f, :l)[1])
    @test tagged(add_tag(γ, 1)[1], add_tag(γ, :l)[1])
    @test tagged(add_tag(b, 1)[1], add_tag(b, :l)[1])
    @test tagged(add_tag(s, 1)[1][:z], add_tag(s, :l)[1][:z])

    # The sorted form doesn't depend on the starting order
    l1, l2, l3 = 2, "a", UInt8(1)
    @test allequal(
        b[x] * b[y] * b[z] for (x, y, z) in
            ((l1, l2, l3), (l1, l3, l2), (l2, l1, l3), (l2, l3, l1), (l3, l1, l2), (l3, l2, l1))
    )
    @test b[1] * b[0] == b[1.0] * b[0]
    @test b["a"] * b[:x] == b[SubString("a")] * b[:x]
end
