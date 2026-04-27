module CombinatoricsExt

using Combinatorics: permutations, parity
import FermionicHilbertSpaces: _resolve_sector_permutations_and_weights, Fill

function _resolve_sector_permutations_and_weights(Hs, sector::Symbol, ::Type{T}) where T
    n = length(Hs)
    n > 0 || throw(ArgumentError("At least one factor space is required"))

    perms = permutations(1:n)
    if sector === :symmetric
        weights = Fill(one(T), length(perms))
        return perms, weights
    elseif sector === :antisymmetric
        weights = Iterators.map(p -> (-1)^parity(p), perms)
        return perms, weights
    end

    throw(ArgumentError("Unknown sector :$(sector). Expected :symmetric or :antisymmetric"))
end

end