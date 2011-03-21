#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <cstdlib>
#include <cstdio>

const int MIN_SIZE = 1;
const int MAX_SIZE = 1048576 * 8;
const int ITERS = 30;

void cpuKernel(void * info)
{
  void * mem = (void * )malloc(MAX_SIZE);
  for (int i = MIN_SIZE; i <= MAX_SIZE; i *= 2)
  {
    dcgn::barrier();
    double t = dcgn::wallTime();
    for (int j = 0; j < ITERS; ++j)
    {
      dcgn::broadcast(0, mem, i);
    }
    t = dcgn::wallTime() - t;
    dcgn::barrier();
    if (dcgn::getRank() == 0) printf("%20.10f\n", t / ITERS);
  }
  free(mem);
}

__global__ void kernel(void * gmem, const dcgn::GPUInitRequest libParam)
{
  dcgn::gpu::init(libParam);
  int index = 0;
  for (int i = MIN_SIZE; i <= MAX_SIZE; i *= 2)
  {
    dcgn::gpu::barrier(0);
    for (int j = 0; j < ITERS; ++j)
    {
      dcgn::gpu::broadcast(0, 0, gmem, i);
    }
    dcgn::gpu::barrier(0);
    ++index;
  }
}

__host__ void gpuKernel(void * info, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  void ** mem = (void ** )info;
  cudaMalloc(mem, MAX_SIZE);
  kernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(*mem, libParam);
}

int main(int argc, char ** argv)
{
  void * gpuMem;
  int gpus[] = { 0, 1, -1 };
  uint3 gs = { 1, 1, 1 }, bs = { 1, 1, 1 };

  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  dcgn::initCPU(2);
  dcgn::initGPU(gpus, 1, 0);
  dcgn::start();

  dcgn::launchCPUKernel(0, cpuKernel, 0);
  dcgn::launchCPUKernel(1, cpuKernel, 0);
  dcgn::launchGPUKernel(0, gpuKernel, 0,  &gpuMem, gs, bs);
  dcgn::launchGPUKernel(1, gpuKernel, 0,  &gpuMem, gs, bs);

  dcgn::finalize();
  return 0;
}
