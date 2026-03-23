## Some types
abstract type AbstractBasisState end
abstract type AbstractFockState <: AbstractBasisState end
abstract type AbstractHilbertSpace{S} end
abstract type AbstractFockHilbertSpace{F<:AbstractFockState} <: AbstractHilbertSpace{F} end
abstract type AbstractAtomicHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractProductHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractClusterHilbertSpace{B} <: AbstractProductHilbertSpace{B} end

factors(H::AbstractAtomicHilbertSpace) = (H,)
clusters(H::AbstractAtomicHilbertSpace) = (H,)
atomic_factors(H::AbstractAtomicHilbertSpace) = (H,)
factors(H::AbstractClusterHilbertSpace) = atomic_factors(H)
clusters(H::AbstractClusterHilbertSpace) = (H,)

partial_trace_phase_factor(f1, f2, ::AbstractAtomicHilbertSpace) = 1

# struct TrivialState <: AbstractBasisState end
# struct TrivialSpace <: AbstractAtomicHilbertSpace{TrivialState} end
# dim(::TrivialSpace) = 1
# basisstates(::TrivialSpace) = (TrivialState(),)
# basisstate(n::Int, ::TrivialSpace) = n == 1 ? TrivialState() : throw(BoundsError())
# state_index(::TrivialState, ::TrivialSpace) = 1
# function Base.show(io::IO, ::TrivialSpace)
#     get(io, :compact, false) ? print(io, "TrivialSpace") : print(io, "TrivialSpace (1-dimensional)")
# end
# complementary_subsystem(H::AbstractHilbertSpace, ::TrivialSpace) = H
# embed(m::UniformScaling, Hsub::TrivialSpace, H::AbstractHilbertSpace; kwargs...) = m.λ * I(dim(H))
# embed(m, Hsub::TrivialSpace, H::AbstractHilbertSpace; kwargs...) = only(m) * I(dim(H))
# _drop_trivial_spaces(spaces) = filter(!Base.Fix2(isa, TrivialSpace), spaces)

# @testitem "TrivialSpace" begin
#     using LinearAlgebra
#     import FermionicHilbertSpaces: TrivialSpace, TrivialState, GenericHilbertSpace, basisstates, basisstate, state_index, factors, clusters, atomic_factors, complementary_subsystem
#     T = TrivialSpace()
#     H = GenericHilbertSpace(:A, [:a, :b])
#     A = [1.0 2.0; 3.0 4.0]
#     α = 2.5
#     mtrivial = fill(α, 1, 1)

#     @test dim(T) == 1
#     @test factors(T) == (T,)
#     @test clusters(T) == (T,)
#     @test atomic_factors(T) == (T,)
#     @test basisstates(T) == (TrivialState(),)
#     @test basisstate(1, T) == TrivialState()
#     @test_throws BoundsError basisstate(2, T)
#     @test state_index(TrivialState(), T) == 1

#     @test complementary_subsystem(H, T) == H
#     @test complementary_subsystem(H, H) == T

#     @test embed(I, T => H) == I(dim(H))
#     @test embed(mtrivial, T => H) == α * I(dim(H))
#     @test embed(A, H => H) == A

#     Hleft = tensor_product((T, H))
#     Hright = tensor_product((H, T))
#     Hleft == Hright == H
#     @test generalized_kron((mtrivial, A), (T, H), Hleft) == α * A
#     @test generalized_kron((A, mtrivial), (H, T), Hright) == α * A

#     @test partial_trace(A, H => H) == A
#     @test partial_trace(embed(A, H => Hright), Hright => H) == A
#     @test partial_trace(embed(mtrivial, T => Hleft), Hleft => T) == fill(α * dim(H), 1, 1)

#     @test matrix_representation(1, T) == I(1)
#     @test matrix_representation(I, T) == I(1)
#     @test matrix_representation(α, T) == α * I(1)

#     @fermions f
#     @test_throws ArgumentError matrix_representation(f[1], T)
# end

