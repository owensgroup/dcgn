#include <dcgn/dcgn.h>
#include <dcgn/CPUWorker.h>

namespace dcgn
{
  bool isCPUIdle(const int localCPUIndex)
  {
    return cpuWorkers[localCPUIndex]->isIdle();
  }
}
