module NonCommutativeProducts

using LinearAlgebra
using TestItems
using BangBang: push!!, pushfirst!!, setindex!!, append!!, mergewith!!
using BangBang.Extras: modify!!

include("mul.jl")
include("add.jl")
include("muladd.jl")
include("sorting.jl")

end
