"""
    parityoperator(H)

Return the fermionic parity operator `(-1)^N` for the Hilbert space `H`: `1` on even and `-1`
on odd basis states.
"""
function parityoperator(H::AbstractHilbertSpace{<:AbstractFockState})
    diagonal_operator(parity, H)
end


"""
    numberoperator(H)

Return the number operator `Σ_j n_j` for the Hilbert space `H`.
"""
function numberoperator(H::AbstractHilbertSpace{<:AbstractFockState})
    diagonal_operator(particle_number, H)
end


diagonal_operator(g, H) = dropzeros!(sparse(Diagonal(map(g, basisstates(H)))))

function togglemajoranas(digitpositions, daggers, focknbr::FockNumber{I}) where {I}
    newfocknbr = zero(focknbr)
    allowed = true
    fermionstatistics = 1
    amp = Complex{Int}(1)
    for (digitpos, dagger) in zip(digitpositions, daggers)
        op = FockNumber(one(I) << (digitpos - 1))
        newfocknbr = op ⊻ focknbr
        if dagger
            occupied = iszero(op & focknbr)
            amp *= 1im * (occupied ? -1 : 1)
        end
        fermionstatistics *= jwstring(digitpos, focknbr)
        focknbr = newfocknbr
    end
    return newfocknbr, allowed * fermionstatistics * amp
end

