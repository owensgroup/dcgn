#ifndef __DCGN_BROADCAST_CU__
#define __DCGN_BROADCAST_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>
#include <dcgn/getRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void broadcast(const int slot, const Target root, void * const buf, const int numBytes)
    {
      volatile GPUIORequest * req = reinterpret_cast<volatile GPUIORequest * >(getRequest(slot));

      req->numBytes = numBytes;
      req->buf = buf;
      req->from = root;
      req->to = slot;
      req->done = 0;
      req->type = REQUEST_TYPE_BROADCAST; // this has to be the last variable set in the structure.
      while (!req->done || req->type != REQUEST_TYPE_NONE) { }
      req->done = 0;
    }
  }
}

#endif
