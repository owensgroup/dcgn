#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>
#include <dcgn/GPUWorker.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>

#include <cstdlib>
#include <cstdio>

namespace dcgn
{
  void start()
  {
    if (mpiWorker == 0)
    {
      fprintf(stderr, "Error, dcgn::start cannot be called before dcgn::init.\n");
      fflush(stderr);
      exit(1);
    }
    profiler->add("Starting DCGN.");
    mpiWorker->start();
    for (int i = 0; i < (int)cpuWorkers.size(); ++i)
    {
      cpuWorkers[i]->setGlobalID(mpiWorker->getTargetForLocalCPU(i));
      cpuWorkers[i]->start();
    }
    for (int i = 0; i < (int)gpuWorkers.size(); ++i)
    {
      gpuWorkers[i]->setLocalID((int)cpuWorkers.size() + i);
      gpuWorkers[i]->setGlobalID(mpiWorker->getTargetForLocalGPU(i, 0));
      gpuWorkers[i]->setPauseTime(mpiWorker->getPauseTime());
      gpuWorkers[i]->start();
    }
  }
}
