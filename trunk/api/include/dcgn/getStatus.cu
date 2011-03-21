#ifndef __DCGN_GETSTATUS_CU__
#define __DCGN_GETSTATUS_CU__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>

namespace dcgn
{
  namespace gpu
  {
    __device__ CommStatus * getStatus(const int slot)
    {
      return gpuStats + slot;
    }
  }
}

#endif
