#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>

#include <cstdlib>
#include <cstdio>

namespace dcgn
{
  void initCPU(const int allocatedCPUs)
  {
    if (mpiWorker == 0)
    {
      fprintf(stderr, "Error, dcgn::initCPU cannot be called before dcgn::init.\n");
      fflush(stderr);
      exit(1);
    }
    profiler->add("Creating CPU worker threads.");
    mpiWorker->setCPUInfo(allocatedCPUs);
    for (int i = 0; i < allocatedCPUs; ++i)
    {
      cpuWorkers.push_back(new CPUWorker(i));
    }
  }
}
