using FermionicHilbertSpaces

@fermions c_l
@fermions c_r

hamiltonian(c) = c[1]'c[1]
jump_op(c) = c[1]'
lindbladian = 1im * (hamiltonian(c_l) - hamiltonian(c_r)) + jump_op(c_l) * jump_op(c_r) + 0.5 * (jump_op(c_l)' * jump_op(c_l) + jump_op(c_r)' * jump_op(c_r))

Hl = hilbert_space(c_l[1])
Hr = hilbert_space(c_r[1])
Hlr = tensor_product((Hl, Hr))
mat = matrix_representation(lindbladian, Hlr)
reshape(mat, Hlr, (Hl, Hr))