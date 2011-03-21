#ifndef __DCGN_GETGPUID_CU__
#define __DCGN_GETGPUID_CU__

#include <dcgn/dcgn.h>
#include <dcgn/init.cu>
#include <dcgn/Request.h>

namespace dcgn
{
  namespace gpu
  {
    __device__ Target getGPUID(const int gpu, const int slot)
    {
      return dcgn_gpu_gpuRanks[gpu] + slot;
    }
  }
}

#endif
