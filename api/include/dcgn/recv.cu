#ifndef __DCGN_RECV_CU__
#define __DCGN_RECV_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>
#include <dcgn/getRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void recv(const int slot, const Target src, void * const buffer, const int numBytes, CommStatus * const stat)
    {
      volatile GPUIORequest * req = reinterpret_cast<volatile GPUIORequest * >(getRequest(slot));

      req->numBytes = numBytes;
      req->buf = buffer;
      req->from = src;
      req->to = slot;
      req->done = 0;
      req->type = REQUEST_TYPE_RECV; // this has to be the last variable set in the structure.
      while (!req->done || req->type != REQUEST_TYPE_NONE) { }
      if (stat) *stat = *const_cast<CommStatus * >(&req->req.stat);
      req->done = 0;
    }
  }
}

#endif
