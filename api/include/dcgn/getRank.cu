#ifndef __DCGN_GETRANK_CU__
#define __DCGN_GETRANK_CU__

#include <dcgn/init.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ Target getRank(const int slot)
    {
      return dcgn_gpu_gpuRank + slot;
    }
  }
}

#endif
