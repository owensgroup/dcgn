#include <dcgn/dcgn.h>
#include <dcgn/CPUWorker.h>
#include <dcgn/GPUWorker.h>

namespace dcgn
{
  Target getRank()
  {
    if      (CPUWorker::isCPUWorkerThread()) return CPUWorker::getRank();
    else if (GPUWorker::isGPUWorkerThread()) return GPUWorker::getRank();
    return static_cast<Target>(-1);
  }
}
