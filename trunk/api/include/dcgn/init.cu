#ifndef __DCGN_INIT_CU__
#define __DCGN_INIT_CU__

#include <dcgn/dcgn.h>
#include <dcgn/OutputStream.cu>
#include <dcgn/Request.h>

__device__ int * dcgn_gpu_cpuRanks;
__device__ int * dcgn_gpu_gpuRanks;
__device__ int dcgn_gpu_garbage;
__device__ int dcgn_gpu_gpuNodeID;
__device__ int dcgn_gpu_numAsyncReqs;
__device__ dcgn::Target dcgn_gpu_gpuRank;
__device__ dcgn::Target dcgn_gpu_gpuSize;
__device__ dcgn::GPUIORequest * dcgn_gpu_gpuReqs;
__device__ dcgn::GPUIORequest * dcgn_gpu_gpuAsyncReqs;
__device__ dcgn::AsyncRequest ** dcgn_gpu_localAsyncReqs;
__device__ unsigned long long * dcgn_gpu_debugInfo;
__device__ int dcgn_gpu_debugInfoIndex;

namespace dcgn
{
  namespace gpu
  {
    __device__ void init(const GPUInitRequest initReq)
    {
      dcgn_gpu_gpuNodeID       = initReq.gpuNodeID;
      dcgn_gpu_gpuRank         = initReq.gpuRank;
      dcgn_gpu_gpuSize         = initReq.gpuSize;
      dcgn_gpu_gpuReqs         = initReq.gpuReqs;
      dcgn_gpu_gpuRanks        = initReq.gpuRanks;
      dcgn_gpu_cpuRanks        = initReq.cpuRanks;
      dcgn_gpu_numAsyncReqs    = initReq.numAsyncReqs;
      dcgn_gpu_gpuAsyncReqs    = initReq.gpuAsyncReqs;
      dcgn_gpu_localAsyncReqs  = initReq.localAsyncReqs;
      dcgn_gpu_gpuOutputStream = initReq.outputStream;
      dcgn_gpu_debugInfo       = initReq.debugInfo;
      dcgn_gpu_debugInfoIndex  = 0;
    }
  }
}

#endif
