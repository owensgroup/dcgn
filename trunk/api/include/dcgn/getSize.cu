#ifndef __DCGN_GETSIZE_CU__
#define __DCGN_GETSIZE_CU__

namespace dcgn
{
  namespace gpu
  {
    __device__ Target getSize()
    {
      return dcgn_gpu_gpuSize;
    }
  }
}

#endif
