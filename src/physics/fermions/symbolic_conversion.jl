struct MajoranaMap{F1,F2}
    ferm_to_maj::F1   # f -> (γ1, γ2)
    maj_to_ferm::F2     # γ -> (f, isfirst::Bool)
end
unpair(f::FermionSym, M::MajoranaMap)::Tuple{MajoranaSym, MajoranaSym} = M.ferm_to_maj(f)
pair(γ::MajoranaSym, M::MajoranaMap)::Tuple{FermionSym, Bool} = M.maj_to_ferm(γ)

majorana_map(; ferm_to_maj, maj_to_ferm) = MajoranaMap(ferm_to_maj, maj_to_ferm)
function majorana_map(H::MajoranaHilbertSpace)
    γ = symbolic_basis(H)
    f = symbolic_fermion_basis(H)
    inv_map = Dict(v => k for (k, v) in mode_ordering(H))
    ferm_to_maj = function (fermion::FermionSym)
        length(fermion.label) == 2 || throw(ArgumentError("Fermion label must be a tuple of two Majorana labels."))
        a, b = fermion.label
        return γ[a], γ[b]
    end
    maj_to_ferm = function (majorana::MajoranaSym)
        pos = _find_position(majorana, H)
        pos == 0 && throw(ArgumentError("Majorana $majorana not found in the Hilbert space."))
        isfirst = isodd(pos)
        paired_pos = isfirst ? pos + 1 : pos - 1
        a = label(majorana)
        b = label(inv_map[paired_pos])
        ferm_label = isfirst ? (a, b) : (b, a)
        return (f[ferm_label], isodd(pos))
    end
    majorana_map(; ferm_to_maj, maj_to_ferm)
end

to_majorana(expr, M::MajoranaMap) = NonCommutativeProducts.ncmap(Base.Fix2(_to_majorana, M), expr)
to_fermion(expr, M::MajoranaMap) = NonCommutativeProducts.ncmap(Base.Fix2(_to_fermion, M), expr)

# leaf functions
function _to_majorana(f::FermionSym, M::MajoranaMap)
    γ1, γ2 = unpair(f, M)
    f.creation ? (1//2 * (γ1 + 1im * γ2)) : 1//2 * (γ1 - 1im * γ2)
end
_to_majorana(x, ::MajoranaMap) = x
function _to_fermion(γ::MajoranaSym, M::MajoranaMap)
    f, isfirst = pair(γ, M)
    isfirst ? f + f' : 1im * (f - f')
end
_to_fermion(x, ::MajoranaMap) = x


function to_fermion(basis::SymbolicMajoranaBasis; name = Symbol("f_", basis.name))
    SymbolicFermionBasis(name, tags(basis).group)
end
function to_majorana(basis::SymbolicFermionBasis; name = Symbol("γ_", basis.name))
    SymbolicMajoranaBasis(name, MajoranaGroup(tags(basis)))
end

@testitem "MajoranaMap (from Hilbert space)" begin
    import FermionicHilbertSpaces: to_majorana, to_fermion, majorana_map, symbolic_fermion_basis

    @majoranas γ
    @test to_majorana(to_fermion(γ); name=:γ) == γ

    H = hilbert_space(γ, 1:4)
    f = symbolic_fermion_basis(H)
    M = majorana_map(H)

    @test to_fermion(γ[1], M) == f[(1, 2)] + f[(1, 2)]'
    @test to_fermion(γ[2], M) == 1im * (f[(1, 2)] - f[(1, 2)]')
    @test to_fermion(2γ[1], M) == 2 * to_fermion(γ[1], M)
    @test to_majorana(1, M) == 1
    @test to_fermion(1, M) == 1

    @test_throws ArgumentError to_fermion(γ[5], M)

    @test to_majorana(to_fermion(γ[1], M), M) == γ[1]
    @test to_fermion(to_majorana(f[(1, 2)], M), M) == f[(1, 2)]

    op = 3 + 2im * γ[1] * γ[3] - 5 * γ[2]
    @test to_majorana(to_fermion(op, M), M) == op
    @test matrix_representation(op, H) == matrix_representation(to_fermion(op, M), parent(H))
end

@testitem "MajoranaMap (custom mapping)" begin
    import FermionicHilbertSpaces: to_majorana, to_fermion, majorana_map

    @majoranas γ
    f = to_fermion(γ; name=:f)

    function maj_to_ferm(majorana)
        lab, tag = majorana.label
        tag == :a ? (f[lab], true) : (f[lab], false)
    end
    ferm_to_maj(fermion) = (γ[fermion.label, :a], γ[fermion.label, :b])

    M = majorana_map(; ferm_to_maj, maj_to_ferm)
    @test to_majorana(f[1], M) == 1//2 * (γ[1, :a] - 1im * γ[1, :b])
    @test to_fermion(γ[1, :a], M) == f[1] + f[1]'
    expr = 2 * f[1] - 3im * f[2]' * f[5] + 5
    @test to_fermion(to_majorana(expr, M), M) == expr
end
