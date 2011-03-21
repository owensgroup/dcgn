#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>

namespace dcgn
{
  void asyncRecv(const Target src,       void * const buffer, const int numBytes, AsyncRequest * const req)
  {
    CPUWorker::asyncRecv(src, buffer, numBytes, req);
  }
}
