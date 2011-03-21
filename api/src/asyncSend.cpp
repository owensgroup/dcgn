#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>

namespace dcgn
{
  void asyncSend(const Target dst, const void * const buffer, const int numBytes, AsyncRequest * const req)
  {
    CPUWorker::asyncSend(dst, buffer, numBytes, req);
  }
}
