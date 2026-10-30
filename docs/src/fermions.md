```@meta
CurrentModule = FermionicHilbertSpaces
```

# Fermionic tensor products, partial traces, and partial transpose

Fermionic operators anticommute, so the tensor product, partial trace and partial transpose of fermionic systems differ from `kron`, the naive partial trace and the ordinary partial transpose. This page explains why and which conventions the package uses, following [[1]](#fermion_information_article) for the tensor product and partial trace and [[2]](#fermionic_partial_time_reversal) for the partial transpose.

## The fermionic tensor product
Creation and annihilation operators satisfy the canonical anticommutation relations ``\{c_i, c_j^\dagger\} = \delta_{ij}``, ``\{c_i, c_j\} = 0``. The Fock space is spanned by occupation number states ``\ket{n_1,\ldots,n_N}`` built by acting with ``c_i^\dagger`` on the vacuum, and is represented as bitstrings of an integer.

Split the modes into two groups ``X`` and ``\bar X`` with Hilbert spaces ``\mathcal{H}_X`` and ``\mathcal{H}_{\bar X}``, and embed local operators ``A`` and ``B`` into ``\mathcal{H} = \mathcal{H}_X \otimes \mathcal{H}_{\bar X}``. The naive choice ``A \otimes \mathbb 1`` and ``\mathbb 1 \otimes B`` always *commutes*, which is inconsistent with the CAR whenever ``A`` and ``B`` are fermionic. A fermion embedded into a larger space must be dressed with a parity operator counting the fermions in the modes it is moved past, a Jordan-Wigner (JW) string. The tensor product that includes this string is the *fermionic tensor product* ``\tilde{\otimes}``, and the embedding ``E_{\mathcal{H}_X \rightarrow \mathcal{H}}[A] = A \tilde{\otimes} \mathbb 1`` is `embed(A, HX => H)`.

!!! note 
    The tensor product depends on the arbitrary ordering of the fermionic modes. **In the special case** where the modes of ``X`` are all to the left of the modes in ``\bar X``, we have the relations 
    ```math
    A \tilde{\otimes} \mathbb 1 = A \otimes \mathbb 1 \\
    \mathbb 1 \tilde{\otimes} B = P \otimes B,
    ```
    where ``P`` is the parity operator on ``\mathcal{H}_X``. 

There are two ways to combine ``A`` and ``B`` into an operator on ``\mathcal{H}``:
1. `tensor_product((A, B), (HX, HXbar) => H)` is the ordered product of embeddings ``A \hat{\otimes} B = (A \tilde{\otimes} \mathbb 1)(\mathbb 1 \tilde{\otimes} B)``, the algebraic product of eq. (26) in [[1]](#fermion_information_article). The order of the arguments matters: swapping two odd operators flips the sign.
2. `generalized_kron((A, B), (HX, HXbar) => H)` is the basis-wise fermionic tensor product ``A \tilde{\otimes} B`` of eqs. (14b) and (15) in [[1]](#fermion_information_article). It equals ``A \hat{\otimes} B`` when each subsystem consists of consecutive modes of ``\mathcal{H}`` passed in that order, and otherwise can differ, for example by a sign; see [`tensor_product` and `generalized_kron` on arrays](hilbert_space_operations.md#tensor_product-and-generalized_kron-on-arrays).

Use `tensor_product` for product states, for products of local observables, and for anything that should equal a product of embedded operators, such as `representation` of a product of symbolic operators.

## The partial trace and partial transpose

The partial trace is most often used to calculate reduced density matrices. The fundamental property is that for an observable ``A`` on the subsystem ``X``, the expectation value of ``A`` in a state ``\rho`` on the full system is equal to the expectation value of ``A`` in the reduced state ``\rho_X = \text{Tr}_{\bar X}[\rho]``.
A more mathematical description is that the partial trace is the adjoint of the embedding: With ``(A|B) = \text{Tr}[A^\dagger B]`` the Hilbert-Schmidt inner product, we have
```math
(E_{\mathcal{H}_X \rightarrow \mathcal{H}}[A] | \rho) = (A | Tr_{\mathcal{H} \rightarrow \mathcal{H}_X}[\rho]).
```
Since the embedding acquires fermionic phases depending on the mode ordering, so does the partial trace. This is handled automatically by `partial_trace(A, H => HX)`.


For bosonic and spin systems, `partial_transpose` is the ordinary matrix transpose on the selected subsystem. For fermions, the ordinary partial transpose does not capture the fermionic structure of the operator algebra. Following [[2]](#fermionic_partial_time_reversal), the package instead implements the fermionic partial transpose, also called partial time reversal.

In the fermionic operator basis, this operation swaps the bra and ket labels on the selected subsystem and multiplies by a phase determined by the fermionic operator content there. Thus it is not, in general, the ordinary matrix transpose in the occupation-number basis. The convention is independent of the ordering of the fermionic modes. Set `phase_factors=false` in `partial_transpose` to disable these phases and obtain the ordinary subsystem transpose.

Unlike the ordinary partial transpose, the fermionic result is not generally Hermitian. If ``p_A`` is the fermion-parity operator on the transposed subsystem, it obeys ``(\rho^{R_A})^\dagger = p_A \rho^{R_A} p_A``. 

## References
```@raw html
<a name="fermion_information_article"></a>
```
[1] Szalay, Szilárd, et al. "Fermionic systems for quantum information people." [Journal of Physics A: Mathematical and Theoretical 54.39 (2021): 393001](https://doi.org/10.1088/1751-8121/ac0646), [arXiv:2006.03087](https://arxiv.org/abs/2006.03087)

```@raw html
<a name="fermionic_partial_time_reversal"></a>
```
[2] H. Shapourian, K. Shiozaki, and S. Ryu, "Partial time-reversal transformation and entanglement negativity in fermionic systems," [Phys. Rev. B 95, 165101 (2017)](https://doi.org/10.1103/PhysRevB.95.165101).
