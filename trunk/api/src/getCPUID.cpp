#include <dcgn/dcgn.h>
#include <dcgn/MPIWorker.h>

namespace dcgn
{
  Target getCPUID(const int cpu)
  {
    return mpiWorker->getTargetForCPU(cpu);
  }
}
