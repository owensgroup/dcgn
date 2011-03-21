#include <dcgn/dcgn.h>
#include <dcgn/MPIWorker.h>

namespace dcgn
{
  __host__ Target getGPUID(const int gpu, const int slot)
  {
    return mpiWorker->getTargetForGPU(gpu, slot);
  }
}
