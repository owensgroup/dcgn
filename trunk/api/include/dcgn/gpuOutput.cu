#ifndef __DCGN_OUTPUT_CU__
#define __DCGN_OUTPUT_CU__

#include <dcgn/dcgn.h>
#include <dcgn/OutputStream.cu>

namespace dcgn
{
  namespace gpu
  {
    __device__ OutputStream & output()
    {
      return *dcgn_gpu_gpuOutputStream;
    }
  }
}

#endif
