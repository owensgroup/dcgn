#ifndef __DCGN_GETREQUEST_CU__
#define __DCGN_GETREQUEST_CU__

#include <dcgn/dcgn.h>
#include <dcgn/init.cu>
#include <dcgn/Request.h>

namespace dcgn
{
  namespace gpu
  {
    __device__ GPUIORequest * getRequest(const int slot)
    {
      return dcgn_gpu_gpuReqs + slot;
    }
  }
}

#endif
