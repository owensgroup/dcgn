#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>

__global__ void gpuKernel(const int gpuID)
{
}

__host__ void gpuKernelWrapper(void * number, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t stream)
{
}
