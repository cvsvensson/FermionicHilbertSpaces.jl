module CombinatoricsExt

using Combinatorics: permutations, parity
using FillArrays: Fill
import FermionicHilbertSpaces: _resolve_sector_permutations_and_weights

function _resolve_sector_permutations_and_weights(Hs, sector::Symbol, ::Type{T}) where T
    n = length(Hs)
    n > 0 || throw(ArgumentError("`Hs` must contain at least one factor space."))

    perms = permutations(1:n)
    if sector === :symmetric
        weights = Fill(one(T), length(perms))
        return perms, weights
    elseif sector === :antisymmetric
        weights = Iterators.map(p -> (-1)^parity(p), perms)
        return perms, weights
    end

    throw(ArgumentError("The sector must be `:symmetric` or `:antisymmetric`, got `:$sector`."))
end

end