#ifndef __DCGN_MALLOC_CU__
#define __DCGN_MALLOC_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>
#include <dcgn/getRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void * malloc(const int slot, const size_t size)
    {
      volatile GPUIORequest * req = reinterpret_cast<volatile GPUIORequest * >(getRequest(slot));
      void * ret;

      req->numBytes = static_cast<int>(size);
      req->buf  = 0;
      req->from = 0;
      req->to   = 0;
      req->done = 0;
      req->type = REQUEST_TYPE_MALLOC; // this has to be the last variable set in the structure.
      while (!req->done || req->type != REQUEST_TYPE_NONE) { }
      ret = req->buf;
      req->buf = 0;
      req->done = 0;

      return ret;
    }
  }
}

#endif
