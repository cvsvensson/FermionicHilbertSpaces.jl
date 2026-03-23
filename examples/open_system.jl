using FermionicHilbertSpaces
using LinearAlgebra
using Test

to_vec(ρ, Hl, Hr, Hlr) = reshape(ρ, (Hl, Hr) => Hlr)
to_mat(v, Hl, Hr, Hlr) = reshape(v, Hlr => (Hl, Hr))

function lindblad_action(mat, ρ, Hl, Hr, Hlr)
    to_mat(mat * to_vec(ρ, Hl, Hr, Hlr), Hl, Hr, Hlr)
end

function normalized_hermitian_matrix(d)
    x = randn(ComplexF64, d, d)
    ρ = x + x'
    ρ ./= tr(ρ)
    ρ
end

@testset "Superoperator vectorization identities" begin
    @fermions c_l
    @fermions c_r

    Hl = hilbert_space(c_l[1])
    Hr = hilbert_space(c_r[1])
    Hlr = tensor_product((Hl, Hr))

    @test dim(Hl) == dim(Hr)

    A = c_l[1]'
    B = c_r[1]
    C = c_l[1]

    MA = matrix_representation(A, Hlr)
    MB = matrix_representation(B, Hlr)
    MC = matrix_representation(C, Hlr)

    Mcomp = matrix_representation(A * B * C, Hlr)
    @test Mcomp ≈ MA * MB * MC

    ρ = randn(ComplexF64, dim(Hl), dim(Hr))
    vρ = to_vec(ρ, Hl, Hr, Hlr)
    @test to_mat(vρ, Hl, Hr, Hlr) ≈ ρ

    @test MA * vρ ≈ to_vec(to_mat(MA * vρ, Hl, Hr, Hlr), Hl, Hr, Hlr)
end

@testset "Lindbladian properties: one mode" begin
    @fermions c_l
    @fermions c_r

    hamiltonian(c) = c[1]' * c[1]
    jump_op(c) = c[1]'
    lindbladian = let Hl = hamiltonian(c_l), Hr = hamiltonian(c_r), Ll = jump_op(c_l), Lr = jump_op(c_r)
        1im * (Hl - Hr) + Ll * Lr - 0.5 * (Ll' * Ll + Lr' * Lr)
    end

    Hl = hilbert_space(c_l[1])
    Hr = hilbert_space(c_r[1])
    Hlr = tensor_product((Hl, Hr))
    mat = matrix_representation(lindbladian, Hlr)

    vals, vecs = eigen(Matrix(mat); sortby=abs)
    stationary_idx = argmin(abs.(vals))
    λss = vals[stationary_idx]
    vss = vecs[:, stationary_idx]

    @test abs(λss) ≤ 1e-10
    @test norm(mat * vss - λss * vss) ≤ 1e-8

    ρ = normalized_hermitian_matrix(dim(Hl))
    dρ = lindblad_action(mat, ρ, Hl, Hr, Hlr)
    @test abs(tr(dρ)) ≤ 1e-10
    @test dρ ≈ dρ'

    ρss = to_mat(vss, Hl, Hr, Hlr)
    @test isfinite(real(tr(ρss)))
end

@testset "Lindbladian properties: two modes, parity sector" begin
    @fermions c_l
    @fermions c_r

    Δ = 1.0
    γ = 0.5
    hamiltonian(c) = Δ * (c[1]' * c[2]' + hc)
    jump_op(c) = c[1] * c[2]

    lindbladian = let Hl = hamiltonian(c_l), Hr = hamiltonian(c_r), Ll = jump_op(c_l), Lr = jump_op(c_r)
        1im * (Hl - Hr) + γ * (Ll * Lr - 0.5 * (Ll' * Ll + Lr' * Lr))
    end

    Hl = hilbert_space(c_l, 1:2, ParityConservation(1))
    Hr = hilbert_space(c_r, 1:2, ParityConservation(1))
    Hlr = tensor_product((Hl, Hr))
    mat = matrix_representation(lindbladian, Hlr)

    @test size(mat) == (dim(Hlr), dim(Hlr))

    ρ = normalized_hermitian_matrix(dim(Hl))
    dρ = lindblad_action(mat, ρ, Hl, Hr, Hlr)
    @test abs(tr(dρ)) ≤ 1e-10
    @test dρ ≈ dρ'

    vals, vecs = eigen(Matrix(mat); sortby=abs)
    stationary_idx = argmin(abs.(real.(vals)))
    λss = vals[stationary_idx]
    vss = vecs[:, stationary_idx]
    @test abs(real(λss)) ≤ 1e-10
    @test norm(mat * vss - λss * vss) ≤ 1e-8

    ρss = to_mat(vss, Hl, Hr, Hlr)
    @test isfinite(real(tr(ρss)))

    ρI = Matrix(I, dim(Hl), dim(Hr))
    vI = to_vec(ρI, Hl, Hr, Hlr)
    @test norm(mat * vI) > 1e-10
end

