#ifndef __DCGN_ASYNCSEND_CU__
#define __DCGN_ASYNCSEND_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>
#include <dcgn/getAsyncRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void asyncSend(const int slot, const Target dst, const void * const buffer, const int numBytes, AsyncRequest * const req)
    {
      volatile GPUIORequest * ioReq = reinterpret_cast<volatile GPUIORequest * >(getAsyncRequest(slot, req));

      ioReq->req.completed = false;
      ioReq->numBytes = numBytes;
      ioReq->buf = const_cast<void * >(buffer);
      ioReq->from = slot;
      ioReq->to = dst;
      ioReq->done = 0;
      ioReq->type = REQUEST_TYPE_SEND; // this has to be the last variable set in the structure.
    }
  }
}

#endif
