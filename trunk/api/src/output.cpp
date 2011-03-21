#include <dcgn/dcgn.h>
#include <dcgn/CPUWorker.h>
#include <dcgn/OutputStream.h>

namespace dcgn
{
  OutputStream & output()
  {
    return CPUWorker::output();
  }
}
