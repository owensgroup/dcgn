#ifndef __DCGN_GETCPUID_CU__
#define __DCGN_GETCPUID_CU__

#include <dcgn/dcgn.h>
#include <dcgn/init.cu>
#include <dcgn/Request.h>

namespace dcgn
{
  namespace gpu
  {
    __device__ Target getCPUID(const int cpu)
    {
      return dcgn_gpu_cpuRanks[cpu];
    }
  }
}

#endif
