#include <dcgn/dcgn.h>
#include <dcgn/CPUWorker.h>
#include <dcgn/GPUWorker.h>

namespace dcgn
{
  bool areAllLocalResourcesIdle()
  {
    for (int i = 0; i < (int)cpuWorkers.size(); ++i) if (!cpuWorkers[i]->isIdle()) return false;
    for (int i = 0; i < (int)gpuWorkers.size(); ++i) if (!gpuWorkers[i]->isIdle()) return false;
    return true;
  }
}
