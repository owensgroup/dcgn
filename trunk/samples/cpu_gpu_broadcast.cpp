#include <dcgn/dcgn.h>
#include <cstdio>
#include <cstdlib>
#include <sched.h>

void gpuKernelWrapper(void * number, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream);

void cpuKernel(void * )
{
  const int id = dcgn::getRank(), size = dcgn::getSize();
  int x;
  if (id == 0)
  {
    x = 22;
    dcgn::broadcast(0, &x, sizeof(int));
    for (int i = 1; i < size; ++i)
    {
      int y;
      dcgn::CommStatus stat;
      dcgn::recv(dcgn::ANY_SOURCE, &y, sizeof(y), &stat);
      x += y;
    }
  }
  else
  {
    dcgn::broadcast(0, &x, sizeof(int));
    dcgn::send(0, &x, sizeof(x));
  }
  dcgn::barrier();
  if (id == 0)
  {
    printf("x = %d\n", x);
  }
}

int main(int argc, char ** argv)
{
  // int cpus[] = { 0 };
  int gpus[] = { 0, 1, -1 };
  uint3 gridSize  = { 1, 1, 1 };
  uint3 blockSize = { 16, 1, 1 };

  dcgn::initAll(&argc, &argv, 2, gpus, 5, 0, -1);

  dcgn::launchCPUKernel(0, cpuKernel,        0);
  dcgn::launchCPUKernel(1, cpuKernel,        0);
  dcgn::launchGPUKernel(0, gpuKernelWrapper, 0, 0, gridSize, blockSize, 0);
  dcgn::launchGPUKernel(1, gpuKernelWrapper, 0, 0, gridSize, blockSize, 0);

  while (!dcgn::areAllLocalResourcesIdle())
  {
    sched_yield();
  }

  dcgn::finalize();

  return 0;
}
