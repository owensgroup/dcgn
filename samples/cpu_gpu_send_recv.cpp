#include <dcgn/dcgn.h>
#include <cstdio>
#include <cstdlib>
#include <sched.h>

void gpuKernelWrapper(void * number, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream);

void cpuKernel(void * )
{
  int id = dcgn::getRank(), x = id, y;
  if (id == 0)
  {
    dcgn::send(id + 1, &x, sizeof(int));
    dcgn::recv(dcgn::ANY_SOURCE,  &x, sizeof(int), 0);
  }
  else if (id == dcgn::getSize() - 1)
  {
    dcgn::recv(id - 1, &y, sizeof(int), 0);
    x += y;
  }
  else
  {
    dcgn::recv(id - 1, &y, sizeof(int), 0);
    x += y;
    dcgn::send(id + 1, &x, sizeof(int));
  }
  dcgn::barrier();
  if (id == 0)
  {
    printf("%s.%s.%d: id = %d, x = %d\n", __FILE__, __FUNCTION__, __LINE__, id, x);
    fflush(stdout);
  }
}

int main(int argc, char ** argv)
{
  // int cpus[] = { 0 };
  int gpus[] = { 0, 1, -1 };
  uint3 gridSize  = { 1, 1, 1 };
  uint3 blockSize = { 1, 1, 1 };

  dcgn::initAll(&argc, &argv, 2, gpus, 1, 0, -1);

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
