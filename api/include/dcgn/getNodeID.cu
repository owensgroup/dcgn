#ifndef __DCGN_GETNODEID_CU__
#define __DCGN_GETNODEID_CU__

#include <dcgn/init.cu>

namespace dcgn
{
  namespace gpu
  {
    int getNodeID()
    {
      return gpuNodeID;
    }
  }
}

#endif
