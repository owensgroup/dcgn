#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>

__global__ void gpuKernel(int * const x, int * const y, const dcgn::GPUInitRequest libParam)
{
  if (threadIdx.x == 0) dcgn::gpu::init(libParam);
  if (threadIdx.x < 5)
  {
    dcgn::gpu::broadcast(threadIdx.x, 0, x, sizeof(int));
    dcgn::gpu::send(threadIdx.x, 0, x, sizeof(int));
    dcgn::gpu::barrier(threadIdx.x);
  }
}

__host__ void gpuKernelWrapper(void * number, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  int * x;
  cudaMalloc(reinterpret_cast<void ** >(&x), sizeof(int) * 2);
  gpuKernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(x, x + 1, libParam);
}
