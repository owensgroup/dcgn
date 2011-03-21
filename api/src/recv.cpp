#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>

namespace dcgn
{
  void recv(const Target src, void * const buffer, const int numBytes, CommStatus * const stat)
  {
    CPUWorker::recv(src, buffer, numBytes, stat);
  }
}
