#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>

__global__ void gpuKernel(const int gpuID, const dcgn::GPUInitRequest libParam)
{
  dcgn::gpu::init(libParam);
  dcgn::gpu::barrier(0);
}

__host__ void gpuKernelWrapper(void * number, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  gpuKernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(0, libParam);
}
