#include <dcgn/dcgn.h>
#include <dcgn/MPIWorker.h>

namespace dcgn
{
  Target getSize()
  {
    return mpiWorker->getGlobalSize();
  }
}
