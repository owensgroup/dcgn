#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>
#include <dcgn/Request.h>

namespace dcgn
{
  void barrier()
  {
    CPUWorker::barrier();
  }
}
