struct NambuState
    state::SingleParticleState
    hole::Bool
end
NambuState(i::Integer, hole::Bool) = NambuState(SingleParticleState(i), hole)

function togglefermions(sites, daggers, f::NambuState)
    (length(sites) == 2 == length(daggers)) || throw(ArgumentError("Must act with exactly two fermions on a NambuState"))
    allowed = sites[2] == only(f.state.sites) && daggers[2] == f.hole
    state = NambuState(SingleParticleState(sites[1]), !daggers[1])
    return state, allowed
end

function normal_order_to_bdg(m::AbstractMatrix)
    T = typeof(one(eltype(m)) / 2)
    mout = similar(m, T)
    mout .= m
    normal_order_to_bdg!(mout)
end
function normal_order_to_bdg!(m::AbstractMatrix)
    n = size(m, 1)
    n == size(m, 2) || throw(ArgumentError("Matrix must be square"))
    h = @views m[1:n÷2, 1:n÷2] / 2
    m[1:n÷2, 1:n÷2] .*= 1 / 2
    m[n÷2+1:end, n÷2+1:end] .= -transpose(h)
    return dropzeros!(m)
end

struct BdGHilbertSpace{H} <: AbstractFockHilbertSpace
    parent::H
    function BdGHilbertSpace(labels)
        states = vec([NambuState(i, hole) for (i, label) in enumerate(labels), hole in (true, false)])
        H = hilbert_space(labels, states)
        return new{typeof(H)}(H)
    end
end
Base.size(h::BdGHilbertSpace) = size(h.parent)
Base.size(h::BdGHilbertSpace, dim) = size(h.parent, dim)
mode_ordering(h::BdGHilbertSpace) = mode_ordering(h.parent)

function matrix_representation(op, H::BdGHilbertSpace)
    isquadratic(op) || throw(ArgumentError("Operator must be quadratic in fermions to be represented on a BdG Hilbert space."))
    normal_order_to_bdg(matrix_representation(remove_identity(op), H.parent))
end

@testitem "BdG" begin
    import FermionicHilbertSpaces: BdGHilbertSpace
    @fermions f
    h = f[1]' * f[2] + 1im * f[1]' * f[2]' + hc
    H = BdGHilbertSpace(1:2)
    @test matrix_representation(h + 1, H) == matrix_representation(h, H)
end