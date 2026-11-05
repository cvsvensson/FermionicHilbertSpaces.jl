module CUDAExt

using CUDA: CUDABackend
using CUDA.cuSPARSE: CuSparseMatrixCSC
import FermionicHilbertSpaces: _device_csc_matrix

_device_csc_matrix(::CUDABackend, colptr, rowval, nzval, dims) = CuSparseMatrixCSC(colptr, rowval, nzval, dims)

end
