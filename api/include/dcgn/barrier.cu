#ifndef __DCGN_BARRIER_CU__
#define __DCGN_BARRIER_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>
#include <dcgn/getRequest.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ void barrier(const int slot)
    {
      volatile GPUIORequest * req = reinterpret_cast<volatile GPUIORequest * >(getRequest(slot));
      req->done = 0;
      req->type = REQUEST_TYPE_BARRIER; // this has to be the last variable set in the structure.
      while (!req->done || req->type != REQUEST_TYPE_NONE) { }
      req->done = 0;
    }
  }
}

#endif
