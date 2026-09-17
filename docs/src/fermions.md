```@meta
CurrentModule = FermionicHilbertSpaces
```

# Fermionic tensor products and partial traces

This page explains the mathematics behind the fermionic tensor product and partial trace, and the conventions this package uses to implement them. See [[1]](#fermion_information_article) for the full formalism; here we only sketch the ideas needed to understand why the package's `tensor_product`, `embed` and `partial_trace` differ from `kron` and a naive partial trace.

### The CAR algebra
Fermionic creation and annihilation operators ``c_i^\dagger, c_i`` satisfy the canonical anticommutation relations (CAR)
```math
\{c_i, c_j^\dagger\} = \delta_{ij}, \quad \{c_i, c_j\} = 0, \quad \{c_i^\dagger, c_j^\dagger\} = 0.
```
The Fock space is spanned by occupation number states ``\ket{n_1,\ldots,n_N}`` built by acting with ``c_i^\dagger`` on the vacuum, and is represented in this package as bitstrings of an integer (see `FockNumber` in [src/physics/fermions/fock.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/blob/main/src/physics/fermions/fock.jl)).

### The fermionic tensor product
Suppose we split the modes into two groups ``X`` and ``\bar X``, with Hilbert spaces ``\mathcal{H}_X`` and ``\mathcal{H}_{\bar X}``, and we want to embed local operators ``A`` on ``\mathcal{H}_X`` and ``B`` on ``\mathcal{H}_{\bar X}`` into the joint space ``\mathcal{H} = \mathcal{H}_X \otimes \mathcal{H}_{\bar X}``. The naive choice ``A \otimes \mathbb 1`` and ``\mathbb 1 \otimes B`` always *commutes*, ``(A \otimes \mathbb 1)(\mathbb 1 \otimes B) = (\mathbb 1 \otimes B)(A \otimes \mathbb 1)``, regardless of what ``A`` and ``B`` are. This is inconsistent with the CAR relations whenever ``A`` and ``B`` involve an odd number of fermionic operators (e.g. single creation/annihilation operators), which must instead *anticommute*.

To embed an operator acting on mode ``i`` into a larger space consistently with the CAR, it must be dressed with a parity operator counting the fermions occupying the modes it is moved past, known as a Jordan-Wigner (JW) string. We refer to the tensor product that includes this JW string as a *fermionic tensor product* and denote it by ``\tilde{\otimes}``. We define the embedding of ``A`` into the joint space as ``E_{\mathcal{H}_X \rightarrow \mathcal{H}}[A] = A \tilde{\otimes} \mathbb 1`` and the code for it is `embed(A, HX => H)`.

!!! note 
    The tensor product depends on the arbitrary ordering of the fermionic modes. **In the special case** where the modes of ``X`` are all to the left of the modes in ``\bar X``, we have the relations 
    ```math
    A \tilde{\otimes} \mathbb 1 = A \otimes \mathbb 1 \\
    \mathbb 1 \tilde{\otimes} B = P \otimes B,
    ```
    where ``P`` is the parity operator on ``\mathcal{H}_X``. 

We still have to define in what order ``A`` and ``B`` acts in the expression ``A \tilde{\otimes} B``. Two choices are reasonable: 
1. decide based on the order we wrote them down, so that ``A \tilde{\otimes} B = - B \tilde{\otimes} A``. 
2. decide based on the arbitrary fermionic mode ordering, in which case ``A \tilde{\otimes} B = B \tilde{\otimes} A``.

In this package, the first choice is the canonical way and is used by `tensor_product`. This function uses the order in which the operators are passed to it explicitly such that ``tensor_product((A, B), (H_X, H_{\bar X}) => H)`` corresponds to the ordered product of embeddings ``A \tilde{\otimes} \mathbb 1) (\mathbb 1 \tilde{\otimes} B)``. The second choice is available with `generalized_kron`.

In this way, our function `tensor_product` is closely related to the `algebraic tensor product` of [[1]](#fermion_information_article).

### The fermionic partial trace
The partial trace is most often used to calculate reduced density matrices. The fundamental property is that for an observable ``A`` on the subsystem ``X``, the expectation value of ``A`` in a state ``\rho`` on the full system is equal to the expectation value of ``A`` in the reduced state ``\rho_X = \text{Tr}_{\bar X}[\rho]``.
A more mathematical description is that the partial trace is the adjoint of the embedding: With ``(A|B) = \text{Tr}[A^\dagger B]`` the Hilbert-Schmidt inner product, we have
```math
(E_{\mathcal{H}_X \rightarrow \mathcal{H}}[A] | \rho) = (A | Tr_{\mathcal{H} \rightarrow \mathcal{H}_X}[\rho]).
```
Since the embedding aquires fermionic phases depending on the mode ordering, so does the partial trace. This is handled automatically by the package when using `partial_trace(A, H => HX)`.

# References
```@raw html
<a name="fermion_information_article"></a>
```
[1] Szalay, Szilárd, et al. "Fermionic systems for quantum information people." [Journal of Physics A: Mathematical and Theoretical 54.39 (2021): 393001](https://doi.org/10.1088/1751-8121/ac0646), [arXiv:2006.03087](https://arxiv.org/abs/2006.03087)
