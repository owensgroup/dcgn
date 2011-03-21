#include <dcgn/dcgn.h>

#include <dcgn/CPUWorker.h>

namespace dcgn
{
  void send(const Target dst, const void * const buffer, const int numBytes)
  {
    CPUWorker::send(dst, buffer, numBytes);
  }
}
