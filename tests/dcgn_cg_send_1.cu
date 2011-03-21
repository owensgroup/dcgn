#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <cstdlib>
#include <cstdio>

const int MIN_SIZE = 1;
const int MAX_SIZE = 1048576;
const int ITERS = 30;

__global__ void kernel(void * gmem, const dcgn::GPUInitRequest libParam)
{
  dcgn::CommStatus stat;
  dcgn::gpu::init(libParam);
  for (int i = MIN_SIZE; i <= MAX_SIZE; i *= 2)
  {
    dcgn::gpu::barrier(0);
    for (int j = 0; j < ITERS; ++j)
    {
      dcgn::gpu::recv(0, 0, gmem, i, &stat);
    }
    dcgn::gpu::barrier(0);
  }
}

__host__ void gpuKernel(void * info, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  void ** mem = (void ** )info;
  cudaMalloc(mem, MAX_SIZE);
  kernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(*mem, libParam);
}

__host__ void gpuDtor(void * info)
{
  cudaFree(*(void ** )info);
}

void cpuKernel(void * info)
{
  void * mem = (void * )malloc(MAX_SIZE);
  for (int i = MIN_SIZE; i <= MAX_SIZE; i *= 2)
  {
    dcgn::barrier();
    double t = dcgn::wallTime();
    for (int j = 0; j < ITERS; ++j)
    {
      dcgn::send(1, mem, i);
    }
    t = dcgn::wallTime() - t;
    dcgn::barrier();
    printf("%10d - %20.10f ms\n", i, t / ITERS * 1000.0f);
  }
  free(mem);
}
#include <mpi.h>
int main(int argc, char ** argv)
{
  void * gpuMem;
  int gpus[] = { 0, -1 };
  uint3 gs = { 1, 1, 1 }, bs = { 1, 1, 1 };

  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  MPI_Barrier(MPI_COMM_WORLD);
  if (dcgn::getNodeID() == 0)
  {
    dcgn::initCPU(1);
    dcgn::initGPU(gpus + 1, 0, 0);
  }
  else
  {
    dcgn::initCPU(0);
    dcgn::initGPU(gpus, 1, 0);
  }
  dcgn::start();

  if (dcgn::getNodeID() == 0) dcgn::launchCPUKernel(0, cpuKernel, 0);
  else                        dcgn::launchGPUKernel(0, gpuKernel, gpuDtor, &gpuMem, gs, bs);

  dcgn::finalize();
  return 0;
}
