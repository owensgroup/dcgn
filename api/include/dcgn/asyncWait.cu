#ifndef __DCGN_ASYNCWAIT_CU__
#define __DCGN_ASYNCWAIT_CU__

#include <dcgn/dcgn.h>
#include <dcgn/getAsyncMatchingRequest.cu>
#include <dcgn/returnAsyncRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void asyncWait(const int slot, AsyncRequest * const req, CommStatus * const stat)
    {
      volatile GPUIORequest * ioReq = reinterpret_cast<volatile GPUIORequest * >(getAsyncMatchingRequest(slot, req));
      if (!ioReq) return;
      while (!ioReq->req.completed) { }
      ioReq->req.completed = false;
      if (stat) *stat = *const_cast<CommStatus * >(&ioReq->req.stat);
      returnAsyncRequest(slot, const_cast<GPUIORequest * >(ioReq));
    }
  }
}

#endif
