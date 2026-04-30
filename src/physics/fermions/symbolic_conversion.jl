struct MajoranaMap{F1,F2}
    ferm_to_maj::F1   # f -> (γ1, γ2)
    maj_to_ferm::F2     # γ -> (f, isfirst::Bool)
end
unpair(f::FermionSym, M::MajoranaMap)::Tuple{MajoranaSym, MajoranaSym} = M.ferm_to_maj(f)
pair(γ::MajoranaSym, M::MajoranaMap)::Tuple{FermionSym, Bool} = M.maj_to_ferm(γ)

majorana_map(; ferm_to_maj, maj_to_ferm) = MajoranaMap(ferm_to_maj, maj_to_ferm)
function majorana_map(H::MajoranaHilbertSpace, f::SymbolicFermionBasis)
    γ = symbolic_basis(H)
    inv_map = Dict(v => k for (k, v) in mode_ordering(H))
    ferm_to_maj = function (fermion::FermionSym)
        length(fermion.label) == 2 || throw(ArgumentError("Fermion label must be a tuple of two Majorana labels."))
        a, b = fermion.label
        return γ[a], γ[b]
    end
    maj_to_ferm = function (majorana::MajoranaSym)
        pos = _find_position(majorana, H)
        pos == 0 && throw(ArgumentError("Majorana $majorana not found in the Hilbert space."))
        paired_pos = isodd(pos) ? pos + 1 : pos - 1
        a = label(majorana)
        b = label(inv_map[paired_pos])
        return (f[(a, b)], isodd(pos))
    end
    majorana_map(; ferm_to_maj, maj_to_ferm)
end

abstract type RewriteDirection end

struct ToMajorana <: RewriteDirection end
struct ToFermion  <: RewriteDirection end

to_majorana(expr, M::MajoranaMap) = _rewrite(expr, M, ToMajorana())
to_fermion(expr, M::MajoranaMap) = _rewrite(expr, M, ToFermion())
function to_majorana(f::FermionSym, M::MajoranaMap)
    γ1, γ2 = unpair(f, M)
    f.creation ? (1//2 * (γ1 - 1im * γ2)) : 1//2 * (γ1 + 1im * γ2)
end
function to_fermion(γ::MajoranaSym, M::MajoranaMap)
    f, isfirst = pair(γ, M)
    isfirst ? f + f' : 1im * (f' - f)
end

_rewrite(expr, ::MajoranaMap, ::RewriteDirection) = expr
_rewrite(expr::FermionSym, M::MajoranaMap, ::ToMajorana) = to_majorana(expr, M)
_rewrite(expr::MajoranaSym, M::MajoranaMap, ::ToFermion) = to_fermion(expr, M)
function _rewrite(expr::NCMul, M::MajoranaMap, dir::RewriteDirection)
    mapreduce(factor -> _rewrite(factor, M, dir), *, expr.factors)
end
function _rewrite(expr::NCAdd, M::MajoranaMap, dir::RewriteDirection)
    sum(_rewrite(term, M, dir) for term in NCterms(expr); init=zero(expr)) + expr.coeff * I
end


@testitem "Fermion to Majorana conversion" begin
    import FermionicHilbertSpaces: to_majorana, to_fermion, majorana_map
    @majoranas γ
    @fermions f
    function maj_to_ferm(majorana)
        lab, tag = majorana.label
        tag == :a ? (f[lab], true) : (f[lab], false)
    end
    ferm_to_maj(fermion) = (γ[fermion.label[1], :a], γ[fermion.label[1], :b])
    M = majorana_map(; ferm_to_maj, maj_to_ferm)
    @test to_majorana(f[1], M) == 1//2 * (γ[1, :a] + 1im * γ[1, :b])
    @test to_fermion(γ[1, :a], M) == f[1] + f[1]'
    @test to_majorana(f[1]' + f[1], M) == γ[1, :a]
    @test to_fermion(1, M) == 1
    @test to_majorana(1, M) == 1

    @test to_majorana(to_fermion(γ[1, :a], M), M) == γ[1, :a]

    H = hilbert_space(γ, 1:4)
    M2 = majorana_map(H, f)
    @test to_fermion(γ[1], M2) == f[(1, 2)] + f[(1, 2)]'

end

