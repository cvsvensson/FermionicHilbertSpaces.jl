module FermionicHilbertSpacesSymbolicsExt

using FermionicHilbertSpaces, Symbolics
import FermionicHilbertSpaces: fermion_to_majorana, majorana_to_fermion, SymbolicMajoranaBasis, SymbolicFermionBasis


function fermion_to_majorana(f::SymbolicFermionBasis, a::SymbolicMajoranaBasis, b::SymbolicMajoranaBasis; leijnse_convention=true)
    a.universe == b.universe || throw(ArgumentError("Majorana bases must anticommute"))
    sgn(x) = leijnse_convention ? (x.creation ? -1 : 1) : (x.creation ? 1 : -1)
    is_fermion_in_basis(x, basis) = x isa FermionicHilbertSpaces.FermionSym && x.basis == basis
    rw = @rule ~x::(x -> is_fermion_in_basis(x, f)) => 1 // 2 * (a[(~x).label] + sgn(~x) * 1im * b[(~x).label])
    return Rewriters.Prewalk(Rewriters.PassThrough(rw))
end

function majorana_to_fermion(a::SymbolicMajoranaBasis, b::SymbolicMajoranaBasis, f::SymbolicFermionBasis; leijnse_convention=true)
    a.universe == b.universe || throw(ArgumentError("Majorana bases must anticommute"))
    sgn = leijnse_convention ? 1 : -1
    is_majorana_in_basis(x, basis) = x isa FermionicHilbertSpaces.MajoranaSym && x.basis == basis
    rw1 = @rule ~x::(x -> is_majorana_in_basis(x, a)) => f[(~x).label] + f[(~x).label]'
    rw2 = @rule ~x::(x -> is_majorana_in_basis(x, b)) => sgn * 1im * (f[(~x).label]' - f[(~x).label])
    return Rewriters.Prewalk(Rewriters.Chain([rw1, rw2]))
end

end
