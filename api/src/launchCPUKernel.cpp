#include <dcgn/dcgn.h>
#include <dcgn/CPUWorker.h>
#include <dcgn/Profiler.h>
#include <cstdlib>
#include <cstdio>

namespace dcgn
{
  void launchCPUKernel(const int cpuThreadIndex, const CPUKernelFunction func, void * const param)
  {
    if (cpuThreadIndex < 0 || cpuThreadIndex >= (int)cpuWorkers.size())
    {
      fprintf(stderr, "Error, trying to launch kernel on invalid CPU worker thread (%d). Aborting.\n", cpuThreadIndex);
      fflush(stderr);
      exit(1);
    }
    profiler->add("Launching CPU kernel.");
    cpuWorkers[cpuThreadIndex]->scheduleKernel(func, param);
    profiler->add("Done.");
  }
}
