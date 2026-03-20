using FermionicHilbertSpaces
using FermionicHilbertSpaces: SpinSpace, TruncatedBosonicHilbertSpace
N = 3
@spins S 1:N
@boson a

ωc = 1.0
ωs = 0.5
λ = 0.1
symham = ωc * a' * a + ωs * sum(S[k][:z] for k in 1:N) + 2λ / sqrt(N) * (a' + a) * sum(S[k][:x] for k in 1:N)

Hs = SpinSpace{1 // 2}.(S)
Ha = TruncatedBosonicHilbertSpace(a, 10)
H = tensor_product(Hs..., Ha)
ham = matrix_representation(symham, H)
# One should use sectors with permutationally invariant states, but we need to add methods to constrain product spaces to do that