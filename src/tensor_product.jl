
"""
    embedding_unitary(partition, fockstates, jw)

    Compute the unitary matrix that maps between the tensor embedding and the fermionic embedding in the physical subspace. 
    # Arguments
    - `partition`: A partition of the labels in `jw` into disjoint sets.
    - `fockstates`: The fock states in the basis
    - `jw`: The Jordan-Wigner ordering.
"""
function embedding_unitary(_partition, fockstates, jw::JordanWignerOrdering)
    #for locally physical algebra, ie only for even operators or states of well-defined parity
    #if ξ is ordered, the phases are +1. 
    # Note that the jordan wigner modes are ordered in reverse from the labels, but this is taken care of by direction of the jwstring below
    partition = map(mode_ordering, _partition)
    isorderedpartition(partition, jw) || throw(ArgumentError("The partition must be ordered according to jw"))

    phases = ones(Int, length(fockstates))
    for (s, Xs) in enumerate(partition)
        mask = focknbr_from_site_labels(Xs, jw)
        for (r, Xr) in Iterators.drop(enumerate(partition), s)
            for li in Xr
                i = siteindex(li, jw)
                for (n, f) in zip(eachindex(phases), fockstates)
                    if _bit(f, i)
                        phases[n] *= jwstring_anti(i, mask & f)
                    end
                end
            end
        end
    end
    return Diagonal(phases)
end

function bipartite_embedding_unitary(_X, _Xbar, fockstates, jw::JordanWignerOrdering)
    #(122a)
    X = mode_ordering(_X)
    Xbar = mode_ordering(_Xbar)
    ispartition((X, Xbar), jw) || throw(ArgumentError("The partition must be ordered according to jw"))
    phases = ones(Int, length(fockstates))
    mask = focknbr_from_site_labels(X, jw)
    for li in Xbar
        i = siteindex(li, jw)
        for (n, f) in zip(eachindex(phases), fockstates)
            if _bit(f, i)
                phases[n] *= jwstring_anti(i, mask & f)
            end
        end
    end
    return Diagonal(phases)
end

@testitem "Embedding unitary" begin
    # Appendix C.4
    import FermionicHilbertSpaces: embedding_unitary, canonical_embedding, bipartite_embedding_unitary
    using LinearAlgebra
    jw = JordanWignerOrdering(1:2)
    fockstates = sort(map(FockNumber, 0:3), by=Base.Fix2(bits, 2))

    @test embedding_unitary([[1], [2]], fockstates, jw) == I
    @test embedding_unitary([[2], [1]], fockstates, jw) == Diagonal([1, 1, 1, -1])

    # N = 3
    jw = JordanWignerOrdering(1:3)
    fockstates = sort(map(FockNumber, 0:7), by=Base.Fix2(bits, 3))
    U(p) = embedding_unitary(p, fockstates, jw)
    @test U([[1], [2], [3]]) == U([[1, 2], [3]]) == U([[1], [2, 3]]) == I

    @test U([[2], [1], [3]]) == Diagonal([1, 1, 1, 1, 1, 1, -1, -1])
    @test U([[2], [3], [1]]) == Diagonal([1, 1, 1, 1, 1, -1, -1, 1])
    @test U([[3], [1], [2]]) == Diagonal([1, 1, 1, -1, 1, -1, 1, 1])
    @test U([[3], [2], [1]]) == Diagonal([1, 1, 1, -1, 1, -1, -1, -1])
    @test U([[1], [3], [2]]) == Diagonal([1, 1, 1, -1, 1, 1, 1, -1])

    @test U([[2], [1, 3]]) == Diagonal([1, 1, 1, 1, 1, 1, -1, -1])
    @test U([[3], [1, 2]]) == Diagonal([1, 1, 1, -1, 1, -1, 1, 1])

    @test U([[1, 3], [2]]) == Diagonal([1, 1, 1, -1, 1, 1, 1, -1])
    @test U([[2, 3], [1]]) == Diagonal([1, 1, 1, 1, 1, -1, -1, 1])
end

"""
    tensor_product(bs)

Compute the tensor_product product of a list of `FermionBasis` objects. The symmetry of the resulting basis is computed by promote_symmetry.
"""
tensor_product(bs::AbstractVector{<:AbstractHilbertSpace}) = foldl(tensor_product, bs)
tensor_product(bs::Tuple) = foldl(tensor_product, bs)

function tensor_product(H1::SymmetricFockHilbertSpace, H2::SymmetricFockHilbertSpace)
    isdisjoint(keys(H1.jw), keys(H2.jw)) || throw(ArgumentError("The labels of the two bases are not disjoint"))
    allequal(isfermionic, (H1, H2)) || throw(ArgumentError("All Hilbert spaces should have the same fermionicity"))
    newlabels = vcat(collect(keys(H1.jw)), collect(keys(H2.jw)))
    qn = promote_symmetry(H1.symmetry, H2.symmetry)
    M1 = length(H1.jw)
    newfocknumbers = vec([f1 + shift_right(f2, M1) for f1 in focknumbers(H1), f2 in focknumbers(H2)])
    SymmetricFockHilbertSpace(newlabels, qn, newfocknumbers; fermionic=isfermionic(H1))
end

function tensor_product(H1::FockHilbertSpace, H2::FockHilbertSpace)
    isdisjoint(keys(H1.jw), keys(H2.jw)) || throw(ArgumentError("The labels of the two bases are not disjoint"))
    allequal(isfermionic, (H1, H2)) || throw(ArgumentError("All Hilbert spaces should have the same fermionicity"))
    newlabels = vcat(collect(keys(H1.jw)), collect(keys(H2.jw)))
    M1 = length(H1.jw)
    newfocknumbers = vec([f1 + shift_right(f2, M1) for f1 in focknumbers(H1), f2 in focknumbers(H2)])
    FockHilbertSpace(newlabels, newfocknumbers; fermionic=isfermionic(H1))
end

function simple_tensor_product(H1::AbstractFockHilbertSpace, H2::AbstractFockHilbertSpace)
    isdisjoint(keys(H1.jw), keys(H2.jw)) || throw(ArgumentError("The labels of the two bases are not disjoint"))
    allequal(isfermionic, (H1, H2)) || throw(ArgumentError("All Hilbert spaces should have the same fermionicity"))
    newlabels = vcat(collect(keys(H1.jw)), collect(keys(H2.jw)))
    SimpleFockHilbertSpace(newlabels; fermionic=isfermionic(H1))
end
tensor_product(H1::SimpleFockHilbertSpace, H2) = simple_tensor_product(H1, H2)
tensor_product(H1, H2::SimpleFockHilbertSpace) = simple_tensor_product(H1, H2)
tensor_product(H1::SimpleFockHilbertSpace, H2::SimpleFockHilbertSpace) = simple_tensor_product(H1, H2)

@testitem "tensor_product product of Fock Hilbert Spaces" begin
    using FermionicHilbertSpaces
    H1 = FockHilbertSpace(1:2)
    H2 = FockHilbertSpace(3:4)
    Hw = tensor_product(H1, H2)
    H3 = FockHilbertSpace(1:4)
    @test Hw == H3
    @test size(H1) .* size(H2) == size(Hw)

    H1 = SymmetricFockHilbertSpace(1:2, FermionConservation())
    H2 = SymmetricFockHilbertSpace(3:4, FermionConservation())
    Hw = tensor_product(H1, H2)
    H3 = SymmetricFockHilbertSpace(1:4, FermionConservation())
    @test Hw == H3
    @test size(H1) .* size(H2) == size(Hw)

    H1 = SymmetricFockHilbertSpace(1:2, ParityConservation())
    H2 = SymmetricFockHilbertSpace(3:4, ParityConservation())
    Hw = tensor_product(H1, H2)
    H3 = SymmetricFockHilbertSpace(1:4, ParityConservation())
    @test Hw == H3
    @test size(H1) .* size(H2) == size(Hw)

end


function check_tensor_product_basis_compatibility(b1::AbstractHilbertSpace, b2::AbstractHilbertSpace, b3::AbstractHilbertSpace)
    if vcat(collect(keys(b1)), collect(keys(b2))) != collect(keys(b3))
        throw(ArgumentError("The labels of the output basis are not the same (or ordered the same) as the labels of the input bases. $(keys(b1)) * $(keys(b2)) != $(keys(b3))"))
    end
end
