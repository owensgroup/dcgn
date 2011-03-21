#ifndef __DCGN_SENDRECVREPLACE_CU__
#define __DCGN_SENDRECVREPLACE_CU__

#include <dcgn/dcgn.h>
#include <dcgn/getRequest.cu>
#include <dcgn/Request.h>

namespace dcgn
{
  namespace gpu
  {
    __device__ void sendRecvReplace(const int slot, const Target dst, const Target src, void * const buffer, const int numBytes, CommStatus * const stat)
    {
      volatile GPUIORequest * req = reinterpret_cast<volatile GPUIORequest * >(getRequest(slot));

      req->numBytes = numBytes;
      req->buf = buffer;
      req->from = src;
      req->to = dst;
      req->done = 0;
      req->type = REQUEST_TYPE_SEND_RECV_REPLACE; // this has to be the last variable set in the structure.
      while (!req->done || req->type != REQUEST_TYPE_NONE) { }
      if (stat) *stat = *const_cast<CommStatus * >(&req->req.stat);
      req->done = 0;
    }
  }
}

#endif
