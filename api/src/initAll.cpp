#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>
#include <dcgn/GPUWorker.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>

#include <cstdlib>

namespace dcgn
{
  void initAll(int * argc, char *** argv, const int allocatedCPUs, const int * const allocatedGPUs, const int numSlotsPerGPU, const int numAsyncGPUTrans, const int pollPauseMS)
  {
    init(argc, argv);
    initComm(pollPauseMS);
    initCPU(allocatedCPUs);
    initGPU(allocatedGPUs, numSlotsPerGPU, numAsyncGPUTrans);
    start();
    /*
    int gpuCount = 0;
    if (allocatedGPUs != 0)
    {
      while (allocatedGPUs[gpuCount] >= 0) { gpuCount++; }
    }
    if (gpuCount > 0 && numSlotsPerGPU <= 0)
    {
      fprintf(stderr, "Error, number of GPU slots must be greater than zero.\n");
      fflush(stderr);
      exit(1);
    }
    profiler = new Profiler;
    // profiler->setEnabled(true);
    profiler->setEnabled(false);
    profiler->setTitle("Master Thread");
    profiler->add("Creating MPI thread.");
    mpiWorker = new MPIWorker(allocatedCPUs, gpuCount, numSlotsPerGPU, pollPauseMS, argc, argv);
    profiler->add("Creating CPU worker threads.");
    for (int i = 0; i < allocatedCPUs; ++i)
    {
      cpuWorkers.push_back(new CPUWorker(i, i, mpiWorker->getTargetForLocalCPU(i)));
    }
    profiler->add("Creating GPU worker threads.");
    for (int i = 0; i < gpuCount; ++i)
    {
      gpuWorkers.push_back(new GPUWorker(numSlotsPerGPU, numAsyncGPUTrans, pollPauseMS, allocatedGPUs[i], i, i + allocatedCPUs, mpiWorker->getTargetForLocalGPU(i, 0)));
    }
    profiler->add("Done creating threads.");
    */
  }
}
