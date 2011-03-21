#ifndef __DCGN_FREE_CU__
#define __DCGN_FREE_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>
#include <dcgn/getRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void free(const int slot, void * const ptr)
    {
      volatile GPUIORequest * req = reinterpret_cast<volatile GPUIORequest * >(getRequest(slot));

      req->numBytes = 0;
      req->buf  = ptr;
      req->from = 0;
      req->to   = 0;
      req->done = 0;
      req->type = REQUEST_TYPE_MALLOC; // this has to be the last variable set in the structure.
      while (!req->done || req->type != REQUEST_TYPE_NONE) { }
      req->done = 0;
    }
  }
}

#endif
