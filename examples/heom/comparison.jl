using LinearAlgebra
using FermionicHilbertSpaces
using FermionicHilbertSpaces: HEOMBosonicBath, heom_bosonic_aux_space, heom_generator

using HierarchicalEOM
using SciMLOperators: concretize
##
# -----------------------------------------------------------------------------
# Shared model parameters (minimal TLS + explicit multi-exponential bosonic bath)
# -----------------------------------------------------------------------------

omega = 1.1
n_exp = 4
m_max = 4

gamma = ComplexF64[0.7+0.0im, 1.3+0.0im, 2, 4, 1im][1:n_exp]
eta = ComplexF64[0.40-0.20im, 0.15+0.05im, 2, 4, 1im][1:n_exp]

# -----------------------------------------------------------------------------
# FermionicHilbertSpaces matrix
# -----------------------------------------------------------------------------

@spin s_l 1 // 2
@spin s_r 1 // 2

H_l = omega * s_l[:z]
H_r = omega * s_r[:z]
V_l = 2s_l[:x]
V_r = 2s_r[:x]

bath_fhs = HEOMBosonicBath(1, n_exp, m_max, gamma, eta)
M_fhs_sym = heom_generator(H_l, H_r, V_l, V_r, bath_fhs)

Hs_l = hilbert_space(s_l)
Hs_r = hilbert_space(s_r)
Haux = heom_bosonic_aux_space(bath_fhs)
Hfull = tensor_product((Hs_l, Hs_r, Haux))

M_fhs = matrix_representation(M_fhs_sym, Hfull)
M_fhs_sys = matrix_representation(1im * (H_l - H_r), tensor_product((Hs_l, Hs_r)))

##
# -----------------------------------------------------------------------------
# HierarchicalEOM matrix
# -----------------------------------------------------------------------------

H_q = 0.5 * omega * sigmaz()
V_q = sigmax()

bath_h = BosonBath(V_q, eta, gamma)
M_h_obj = M_Boson(H_q, m_max, bath_h; threshold=0.0, verbose=false)
M_h = concretize(M_h_obj.data)

M_h_sys = concretize(M_S(H_q; verbose=false).data)
println("FHS matrix size:  ", size(M_fhs))
println("HEOM matrix size: ", size(M_h))
@assert size(M_fhs) == size(M_h) "Matrix dimensions differ; cannot compare"
##
M_h - transpose(M_fhs)
