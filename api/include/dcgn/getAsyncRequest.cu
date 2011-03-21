#ifndef __DCGN_GETASYNCREQUEST_CU__
#define __DCGN_GETASYNCREQUEST_CU__

#include <dcgn/dcgn.h>
#include <dcgn/init.cu>
#include <dcgn/Request.h>

namespace dcgn
{
  namespace gpu
  {
    __device__ GPUIORequest * getAsyncRequest(const int slot, AsyncRequest * const asyncReq)
    {
      const int num = dcgn_gpu_numAsyncReqs;
      AsyncRequest ** const reqs = dcgn_gpu_localAsyncReqs + slot * num;
      dcgn_gpu_debugInfo[dcgn_gpu_debugInfoIndex++] = 0x100;
      while (true)
      {
        for (int i = 0; i < num; ++i)
        {
          if (reqs[i] == reinterpret_cast<AsyncRequest * >(0xDeadBeef)) // yes yes, incredibly annoying that we can't just use null.
          {
            // debugInfo[debugInfoIndex++] = 0xBeefDead;
            // debugInfo[debugInfoIndex++] = (unsigned long long)reqs[i];
            // debugInfo[debugInfoIndex++] = i;
            reqs[i] = asyncReq;
            return dcgn_gpu_gpuAsyncReqs + (slot * num + i);
          }/*
          else
          {
            debugInfo[debugInfoIndex++] = 0xDeadFeeb;
            debugInfo[debugInfoIndex++] = (unsigned long long)reqs[i];
            debugInfo[debugInfoIndex++] = i;
            reqs[i] = asyncReq;
            return gpuAsyncReqs + slot * num + i;
          }*/
        }
      }
    }
  }
}

#endif
