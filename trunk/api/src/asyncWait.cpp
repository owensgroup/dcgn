#include <dcgn/dcgn.h>
#include <dcgn/Profiler.h>
#include <dcgn/Thread.h>

namespace dcgn
{
  void asyncWait(AsyncRequest * const req, CommStatus * const stat)
  {
    if (!req) return;
    profiler->add("Waiting for asynchronous request.");
    while (!req->completed)
    {
      Thread::yield();
    }
    if (stat) *stat = req->stat;
    profiler->add("Done waiting for asynchronous request.");
  }
}
