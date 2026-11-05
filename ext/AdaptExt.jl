module AdaptExt

using Adapt: Adapt
using FermionicHilbertSpaces: GatherMap, ReshapeMap

Adapt.adapt_structure(to, op::GatherMap) = GatherMap(op.f, Adapt.adapt(to, op.args))
Adapt.adapt_structure(to, op::ReshapeMap) =
    ReshapeMap(op.mappings, op.repeat, op.projection, map(p -> merge(p, (; blocks = Adapt.adapt(to, p.blocks))), op.precomputed))

end
