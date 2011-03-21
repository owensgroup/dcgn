#include <dcgn/dcgn.h>

#include <dcgn/GPUWorker.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>

#include <cstdlib>
#include <cstdio>

namespace dcgn
{
  void initGPU(const int * const allocatedGPUs, const int slotsPerGPU, const int asyncTransfersPerSlot)
  {
    int gpuCount = 0;
    if (allocatedGPUs != 0)
    {
      while (allocatedGPUs[gpuCount] >= 0) { gpuCount++; }
    }
    if (mpiWorker == 0)
    {
      fprintf(stderr, "Error, dcgn::initGPU cannot be called before dcgn::init.\n");
      fflush(stderr);
      exit(1);
    }
    profiler->add("Creating GPU worker threads.");
    mpiWorker->setGPUInfo(gpuCount, slotsPerGPU);
    for (int i = 0; i < gpuCount; ++i)
    {
      gpuWorkers.push_back(new GPUWorker(slotsPerGPU, asyncTransfersPerSlot, allocatedGPUs[i], i));
    }
  }
}
