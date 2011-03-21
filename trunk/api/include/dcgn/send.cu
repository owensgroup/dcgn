#ifndef __DCGN_SEND_CU__
#define __DCGN_SEND_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>
#include <dcgn/getRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void send(const int slot, const Target dst, const void * const buffer, const int numBytes)
    {
      volatile GPUIORequest * req = reinterpret_cast<volatile GPUIORequest * >(getRequest(slot));

      req->numBytes = numBytes;
      req->buf = const_cast<void * >(buffer);
      req->from = slot;
      req->to = dst;
      req->done = 0;
      req->type = REQUEST_TYPE_SEND; // this has to be the last variable set in the structure.
      while (!req->done || req->type != REQUEST_TYPE_NONE) { }
      req->done = 0;
    }
  }
}

#endif
