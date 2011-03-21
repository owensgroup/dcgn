#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <cstdio>

const int NUM_COMM = 10;

typedef struct _KernelInfo
{
  int * x;
  int id;
} KernelInfo;

__global__ void kernel(dcgn::GPUInitRequest libParam, int * x);

__host__ void gpuKernel(void * kernelInfo, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  KernelInfo * kinfo = reinterpret_cast<KernelInfo * >(kernelInfo);
  cudaMalloc(reinterpret_cast<void ** >(&kinfo->x), sizeof(int) * NUM_COMM * 2);
  cudaMemset(kinfo->x, 0, sizeof(int) * NUM_COMM * 2);
  kinfo->id = libParam.gpuRank;
  kernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(libParam, kinfo->x);
}
__global__ void kernel(dcgn::GPUInitRequest libParam, int * x)
{
  dcgn::gpu::init(libParam);
  dcgn::AsyncRequest reqs[NUM_COMM];
  if (dcgn::gpu::getRank(0) == 0)
  {
    for (int i = 0; i < NUM_COMM; ++i) x[i] = i + 1;
    for (int i = 0; i < NUM_COMM; ++i)
    {
      dcgn::gpu::asyncSend(0, 1, x + i, sizeof(int), reqs + i);
    }
  }
  else
  {
    for (int i = 0; i < NUM_COMM; ++i)
    {
      dcgn::gpu::asyncRecv(0, 0, x + i, sizeof(int), reqs + i);
    }
  }
  for (int i = 0; i < NUM_COMM; ++i)
  {
    dcgn::CommStatus stat;
    dcgn::gpu::asyncWait(0, reqs + i, &stat);
  }
  if (dcgn::gpu::getRank(0) == 0)
  {
    for (int i = 0; i < NUM_COMM; ++i) x[NUM_COMM + i] = i + 1;
    for (int i = 0; i < NUM_COMM; ++i)
    {
      dcgn::gpu::asyncSend(0, 1, x + NUM_COMM + i, sizeof(int), reqs + i);
    }
  }
  else
  {
    for (int i = 0; i < NUM_COMM; ++i)
    {
      dcgn::gpu::asyncRecv(0, 0, x + NUM_COMM + i, sizeof(int), reqs + i);
    }
  }
  for (int i = 0; i < NUM_COMM; ++i)
  {
    dcgn::CommStatus stat;
    dcgn::gpu::asyncWait(0, reqs + i, &stat);
  }
}

void gpuDtor(void * kernelInfo)
{
  KernelInfo * kinfo = reinterpret_cast<KernelInfo * >(kernelInfo);
  if (kinfo->id == 1)
  {
    int * x = new int[NUM_COMM * 2];
    cudaMemcpy(x, kinfo->x, sizeof(int) * NUM_COMM * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_COMM * 2; ++i)
    {
      printf("%2d: %2d\n", i, x[i]);
    }
    delete [] x;
  }
  cudaFree(kinfo->x);
}

int main(int argc, char ** argv)
{
  int gpus[] = { 0, 1, -1 };
  KernelInfo k1, k2;
  uint3 gs = { 1, 1, 1 }, bs = { 1, 1, 1 };

  dcgn::initAll(&argc, &argv, 0, gpus, 1, NUM_COMM, 100);

  dcgn::launchGPUKernel(0, gpuKernel, 0,        reinterpret_cast<void * >(&k1), gs, bs);
  dcgn::launchGPUKernel(1, gpuKernel, gpuDtor,  reinterpret_cast<void * >(&k2), gs, bs);

  dcgn::finalize();

  return 0;
}
