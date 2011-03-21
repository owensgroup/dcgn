#include <dcgn/dcgn.h>
#include <dcgn/MPIWorker.h>

namespace dcgn
{
  int getNodeID()
  {
    return mpiWorker->getMPIRank();
  }
}
