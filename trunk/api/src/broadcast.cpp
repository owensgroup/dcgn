#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>
#include <dcgn/Request.h>

namespace dcgn
{
  void broadcast(const Target root, void * const bytes, const int numBytes)
  {
    CPUWorker::broadcast(root, bytes, numBytes);
  }
}
