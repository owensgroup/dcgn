#include <dcgn/dcgn.h>
#include <dcgn/GPUWorker.h>

namespace dcgn
{
  bool isGPUIdle(const int localGPUIndex)
  {
    return gpuWorkers[localGPUIndex]->isIdle();
  }
}
