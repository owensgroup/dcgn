#include <mpi.h>

#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>
#include <dcgn/GPUWorker.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>

#include <cstdio>

namespace dcgn
{
  void finalize()
  {
    profiler->add("Shutting down cpu threads.");
    for (int i = 0; i < (int)cpuWorkers.size(); ++i) cpuWorkers[i]->shutdown();
    profiler->add("Shutting down gpu threads.");
    for (int i = 0; i < (int)gpuWorkers.size(); ++i) gpuWorkers[i]->shutdown();

    for (int i = 0; i < (int)cpuWorkers.size(); ++i)
    {
      profiler->add("Finalizing cpu thread %d.", i);
      cpuWorkers[i]->waitForShutdown();
      delete cpuWorkers[i];
    }
    for (int i = 0; i < (int)gpuWorkers.size(); ++i)
    {
      profiler->add("Finalizing gpu thread %d.", i);
      gpuWorkers[i]->waitForShutdown();
      delete gpuWorkers[i];
    }
    mpiWorker->shutdown();
    mpiWorker->waitForShutdown();
    for (int i = 0; i < mpiWorker->getMPISize(); ++i)
    {
      for (int j = 0; j < 1000; ++j) MPI_Barrier(MPI_COMM_WORLD);
      if (i == mpiWorker->getMPIRank())
      {
        fprintf(stderr, "%s", profiler->getAllEvents().c_str());
        fflush(stderr);
        // fprintf(stderr, "%s", profiler->getTimes().c_str());
        // fflush(stderr);
      }
    }
    delete mpiWorker;
  }
}
