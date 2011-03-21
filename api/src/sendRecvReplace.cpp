#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>

namespace dcgn
{
  void sendRecvReplace(const Target dst, const Target src, void * const buffer, const int numBytes, CommStatus * const stat)
  {
    CPUWorker::sendRecvReplace(dst, src, buffer, numBytes, stat);
  }
}
