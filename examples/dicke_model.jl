using FermionicHilbertSpaces

N = 3
@spins S 1:N
@boson a

ωc = 1.0
ωz = 0.5
λ = 0.1
symham = ωc * a' * a + ωz * sum(S[k][:z] for k in 1:N) + 2λ / sqrt(N) * (a' + a) * sum(S[k][:x] for k in 1:N)

Hs = hilbert_space.(values(S), 1//2)
Ha = hilbert_space(a, 10)
H = tensor_product(Hs..., Ha, ParityConservation())
ham = matrix_representation(symham, H)
# One should use sectors with permutationally invariant states, but we need to add methods to constrain product spaces to do that
