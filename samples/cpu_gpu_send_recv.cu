#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>

__global__ void gpuKernel(int * const x, int * const y, const dcgn::GPUInitRequest libParam)
{
  dcgn::gpu::init(libParam);
  int id = dcgn::gpu::getRank(0);
  *x = id;
  if (id == 0)
  {
    dcgn::gpu::send(0, id + 1,            x, sizeof(int));
    dcgn::gpu::recv(0, dcgn::ANY_SOURCE, x, sizeof(int), 0);
  }
  else if (id == dcgn::gpu::getSize() - 1)
  {
    dcgn::gpu::recv(0, dcgn::ANY_SOURCE, y, sizeof(int), 0);
    *x += *y;
    dcgn::gpu::send(0, 0,                 x, sizeof(int));
  }
  else
  {
    dcgn::gpu::recv(0, dcgn::ANY_SOURCE, y, sizeof(int), 0);
    *x += *y;
    dcgn::gpu::send(0, id + 1,            x, sizeof(int));
  }
  dcgn::gpu::barrier(0);
}

__host__ void gpuKernelWrapper(void * number, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  int * x;
  cudaMalloc(reinterpret_cast<void ** >(&x), sizeof(int) * 2);
  gpuKernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(x, x + 1, libParam);
}
