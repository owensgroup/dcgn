#ifndef __DCGN_RETURNASYNCREQUEST_CU__
#define __DCGN_RETURNASYNCREQUEST_CU__

#include <dcgn/dcgn.h>
#include <dcgn/init.cu>
#include <dcgn/Request.h>

namespace dcgn
{
  namespace gpu
  {
    __device__ void returnAsyncRequest(const int slot, GPUIORequest * const ioReq)
    {
      // debugInfo[debugInfoIndex++] = 0;
      // debugInfo[debugInfoIndex++] = ioReq - gpuAsyncReqs;

      // why in the heck do we need a scatter to make this stuff work right? when a series of send/wait is called
      // repeatedly, asyncWait doesn't read something correctly. i *think* what isn't working is the assignment
      // of localAsyncReqs[index] in this function. for some reason, when i add a scatter call to this function,
      // everything starts working...
      dcgn_gpu_debugInfo[0] = ioReq - dcgn_gpu_gpuAsyncReqs;
      dcgn_gpu_localAsyncReqs[ioReq - dcgn_gpu_gpuAsyncReqs] = reinterpret_cast<AsyncRequest * >(0xDeadBeef);
    }
  }
}

#endif
