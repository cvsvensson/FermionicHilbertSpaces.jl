"""
    removefermion(digitposition, f::FockNumber)

Return (newfocknbr, fermionstatistics) where `newfocknbr` is the state obtained by removing a fermion at `digitposition` from `f` and `fermionstatistics` is the phase from the Jordan-Wigner string, or 0 if the operation is not allowed.
"""
function removefermion(digitposition, f::FockNumber)
    cdag = focknbr_from_site_index(digitposition)
    newfocknbr = cdag ⊻ f
    allowed = !iszero(cdag & f)
    fermionstatistics = jwstring(digitposition, f)
    return allowed * newfocknbr, allowed * fermionstatistics
end

"""
    parityoperator(H)

Return the fermionic parity operator for the Hilbert space `H`.
"""
function parityoperator(H::AbstractHilbertSpace{<:AbstractFockState})
    sparse_fockoperator(f -> (f, parity(f)), H)
end


"""
    numberoperator(H)

Return the number operator for the Hilbert space `H`.
"""
function numberoperator(H::AbstractHilbertSpace{<:AbstractFockState})
    sparse_fockoperator(f -> (f, fermionnumber(f)), H)
end


"""
    togglefermions(digitpositions, daggers, f::FockNumber)

Return (newfocknbr, fermionstatistics) where `newfocknbr` is the state obtained by toggling fermions at `digitpositions` with `daggers` in the Fock state `f`, and `fermionstatistics` is the phase from the Jordan-Wigner string. If the operation puts two fermions one the same site, the resulting state is undefined.
"""
function togglefermions(digitpositions, daggers, focknbr::FockNumber{I}) where I
    newfocknbr = focknbr
    fermionstatistics = 1
    for (dagger, digitpos) in zip(daggers, digitpositions)
        op = one(I) << (digitpos - 1)
        occupied = !iszero(op & newfocknbr)
        if dagger == occupied  # creation on occupied OR annihilation on empty
            return newfocknbr, 0
        end
        fermionstatistics *= jwstring(digitpos, newfocknbr)
        # Apply operator: both cases reduce to XOR
        # creation:     0|op  XOR focknbr sets the bit (since bit was 0)
        # annihilation: op    XOR focknbr clears the bit (since bit was 1)
        newfocknbr = op ⊻ newfocknbr
    end
    return newfocknbr, fermionstatistics
end

function togglemajoranas(digitpositions, daggers, focknbr)
    newfocknbr = zero(focknbr)
    allowed = true
    fermionstatistics = 1
    amp = Complex{Int}(1)
    for (digitpos, dagger) in zip(digitpositions, daggers)
        op = FockNumber(1 << (digitpos - 1))
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

