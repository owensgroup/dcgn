#ifndef __DCGN_ASYNCRECV_CU__
#define __DCGN_ASYNCRECV_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>
#include <dcgn/getAsyncRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void asyncRecv(const int slot, const Target src, void * const buffer, const int numBytes, AsyncRequest * const req)
    {
      volatile GPUIORequest * ioReq = reinterpret_cast<volatile GPUIORequest * >(getAsyncRequest(slot, req));

      ioReq->req.completed = false;
      ioReq->numBytes = numBytes;
      ioReq->buf = buffer;
      ioReq->from = src;
      ioReq->to = slot;
      ioReq->done = 0;
      ioReq->type = REQUEST_TYPE_RECV; // this has to be the last variable set in the structure.
    }
  }
}

#endif
