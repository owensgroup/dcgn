#include <dcgn/dcgn.h>
#include <dcgn/GPUWorker.h>
#include <dcgn/Profiler.h>

#include <cstdio>

namespace dcgn
{
  void launchGPUKernel(const int gpuThreadIndex, const GPUKernelFunction func, const GPUCleanupFunction dtor, void * const param, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize)
  {
    if (gpuThreadIndex < 0 || gpuThreadIndex >= (int)gpuWorkers.size())
    {
      fprintf(stderr, "Error, trying to launch kernel on invalid GPU worker thread (%d). Aborting.\n", gpuThreadIndex);
      fflush(stderr);
      exit(1);
    }
    profiler->add("Launching GPU kernel.");
    gpuWorkers[gpuThreadIndex]->scheduleKernel(func, dtor, param, gridSize, blockSize, sharedMemSize);
    profiler->add("Done.");
  }
}
