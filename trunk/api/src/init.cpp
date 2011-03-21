#include <dcgn/dcgn.h>

#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>

#include <cstdlib>

namespace dcgn
{
  void init(int * argc, char *** argv)
  {
    profiler = new Profiler;
    profiler->setEnabled(false);
    profiler->setTitle("Master Thread");

    profiler->add("Creating MPI worker thread.");
    mpiWorker = new MPIWorker(argc, argv);
/*
    if (gpuCount > 0 && numSlotsPerGPU <= 0)
    {
      fprintf(stderr, "Error, number of GPU slots must be greater than zero.\n");
      fflush(stderr);
      exit(1);
    }
    profiler->add("Creating MPI thread.");
    profiler->add("Creating GPU worker threads.");
    for (int i = 0; i < gpuCount; ++i)
    {
      gpuWorkers.push_back(new GPUWorker(numSlotsPerGPU, numAsyncGPUTrans, pollPauseMS, allocatedGPUs[i], i, i + allocatedCPUs, mpiWorker->getTargetForLocalGPU(i, 0)));
    }
    profiler->add("Done creating threads.");
*/
  }
}
