```@meta
CurrentModule = FermionicHilbertSpaces
```

# Misc

## No double occupation

The following example demonstrates how to restrict the Hilbert space to only allow single occupation of a site, which can be useful for simulating systems where the on-site coulomb interaction is strong enough to prevent double occupation.

```@example intro
using FermionicHilbertSpaces, LinearAlgebra
N = 2 # number of fermions
space = 1:N 
spin = (:↑,:↓)
# labels = Base.product(space, spin) 
Hs = [hilbert_space([(k, s) for s in spin], FermionConservation(0:1)) for k in space]
H = tensor_product(Hs)
```

```@example intro
size(H,1) == 3^N
```

Can also take product of symmetries
```@example intro
qn2 = prod(IndexConservation(k,0:1) for k in space)
H2 = hilbert_space(keys(H), qn2)
```

This gives the same states but with a different ordering.
```@example intro
sort(focknumbers(H2), by = f->f.f) == sort(focknumbers(H), by = f->f.f)
```

