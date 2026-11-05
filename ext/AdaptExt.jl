module AdaptExt

using Adapt: Adapt, adapt
using FermionicHilbertSpaces: PartialTraceMap, EmbedMap, ReshapeMap, _PartialTraceGather, _EmbedGather, _GatherBlock

Adapt.adapt_structure(to, g::_PartialTraceGather) = _PartialTraceGather(adapt(to, g.L), adapt(to, g.W))
Adapt.adapt_structure(to, g::_EmbedGather) = _EmbedGather(adapt(to, g.P), adapt(to, g.B), adapt(to, g.S))
Adapt.adapt_structure(to, op::PartialTraceMap{<:Any, <:Any, <:Any, <:Any, <:_PartialTraceGather}) = PartialTraceMap(op.H, op.Hsub, op.complement, op.kwargs, adapt(to, op.map))
Adapt.adapt_structure(to, op::EmbedMap{<:Any, <:Any, <:Any, <:Any, <:_EmbedGather}) = EmbedMap(op.Hsub, op.H, op.complement, op.kwargs, adapt(to, op.map))

# Only the tables of gather blocks move: the dropped indices are used on the host, and generic blocks are scalar-indexed.
_adapt_block(to, b::_GatherBlock) = merge(b, (; L = adapt(to, b.L), W = adapt(to, b.W)))
_adapt_block(to, b) = b
Adapt.adapt_structure(to, op::ReshapeMap) =
    ReshapeMap(op.mappings, op.repeat, op.projection, map(p -> merge(p, (; blocks = _adapt_block(to, p.blocks))), op.precomputed))

end
