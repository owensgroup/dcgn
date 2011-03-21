#ifndef __DCGN_GETASYNCMATCHINGREQUEST_CU__
#define __DCGN_GETASYNCMATCHINGREQUEST_CU__

#include <dcgn/dcgn.h>
#include <dcgn/init.cu>
#include <dcgn/Request.h>

namespace dcgn
{
  namespace gpu
  {
    __device__ GPUIORequest * getAsyncMatchingRequest(const int slot, AsyncRequest * const asyncReq)
    {
      const int num = dcgn_gpu_numAsyncReqs;
      AsyncRequest ** const reqs = dcgn_gpu_localAsyncReqs + slot * num;
      for (int i = 0; i < num; ++i)
      {
        if (reqs[i] == asyncReq) return dcgn_gpu_gpuAsyncReqs + slot * num + i;
      }
      return 0;
    }
  }
}

#endif
