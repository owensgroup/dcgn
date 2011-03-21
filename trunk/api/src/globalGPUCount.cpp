#include <dcgn/dcgn.h>
#include <dcgn/MPIWorker.h>

namespace dcgn
{
  int globalGPUCount()
  {
    return mpiWorker->getGlobalGPUCount();
  }
}

