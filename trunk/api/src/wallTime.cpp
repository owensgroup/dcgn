#include <dcgn/dcgn.h>
#include <dcgn/MPIWorker.h>

namespace dcgn
{
  double wallTime()
  {
    return mpiWorker->wallTime();
  }
}
