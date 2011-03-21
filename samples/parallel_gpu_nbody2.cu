#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include "samples/Body.cxx"
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define BARRIER_BEFORE_BIG_COMM 1

#define CHECK_ERR()                                                                               \
{                                                                                                 \
  cudaError_t err = cudaGetLastError();                                                           \
  if (err != cudaSuccess)                                                                         \
  {                                                                                               \
    fprintf(stderr, "%s.%s.%d: %s\n", __FILE__, __FUNCTION__, __LINE__, cudaGetErrorString(err)); \
    fflush(stderr);                                                                               \
    exit(1);                                                                                      \
  }                                                                                               \
}                                                                                                 \

typedef struct _KernelInfo
{
  int n, timeSteps;
  char * input, * output;
  float timeDelta;
} KernelInfo;

__device__ void block_memcpy(void * dstMem, const void * srcMem, const int size);
__device__ void block_memcpy2(void * dstMem, const void * srcMem, const int size);
__device__ void updateForces(Body * bodies, Body * ubodies, const int startBody, const int endBody, const int n, const float timeDelta);
__host__ void runKernel(Body * bodies, Body * gpuBodies, Body * gpuUpdatedBodies, const int n, const int timeSteps, const float timeDelta);
__host__ void nbodyWrapper(void * kernelInfo, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream);
void nbodyTimer(void * kernelInfo);
__global__ void nbody(dcgn::GPUInitRequest libParam, Body * gpuBodies, Body * gpuUpdatedBodies, volatile int * sbarr, const int size, const int timeSteps, const float timeDelta);
__device__ __host__ int getIndexForRank(const int rank, const int size, const int n);
__device__ void __syncblocks(volatile int * sbarr);

int main(int argc, char ** argv)
{
  int gpus[] = { 0, 1, -1 };
  uint3 gs = { 12, 1, 1 }, bs = { 128, 1, 1 };

  dcgn::initAll(&argc, &argv, 1, gpus, 1, 0, -1);

  if (argc != 5)
  {
    dcgn::finalize();
    return 0;
  }
  FILE * fp = fopen(argv[1], "rb");
  KernelInfo kinfo;
  fread(&kinfo.n, sizeof(int), 1, fp);
  fread(&kinfo.timeSteps, sizeof(int), 1, fp);
  fclose(fp);

  kinfo.input = argv[1];
  kinfo.output = argv[2];

  sscanf(argv[3], "%d", &kinfo.timeSteps);
  sscanf(argv[4], "%f", &kinfo.timeDelta);

  dcgn::launchCPUKernel(0, nbodyTimer,  reinterpret_cast<void * >(&kinfo));
  dcgn::launchGPUKernel(0, nbodyWrapper,  0, reinterpret_cast<void * >(&kinfo), gs, bs);
  dcgn::launchGPUKernel(1, nbodyWrapper,  0, reinterpret_cast<void * >(&kinfo), gs, bs);

  dcgn::finalize();

  return 0;
}

__host__ void nbodyWrapper(void * kernelInfo, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  KernelInfo * const kinfo = reinterpret_cast<KernelInfo * >(kernelInfo);
  Body * gpuBodies, * gpuUpdatedBodies;
  volatile int * sbarr;

  cudaMalloc(reinterpret_cast<void ** >(const_cast<int ** >(&sbarr)), sizeof(int) * gridSize.x);  CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuBodies),                  sizeof(Body) * kinfo->n);   CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuUpdatedBodies),           sizeof(Body) * kinfo->n);   CHECK_ERR();
  cudaMemset(const_cast<int * >(sbarr), 0, sizeof(int) * gridSize.x);                             CHECK_ERR();

  nbody<<<gridSize, blockSize, sharedMemSize, *stream>>>(libParam, gpuBodies, gpuUpdatedBodies, sbarr, kinfo->n, kinfo->timeSteps, kinfo->timeDelta); CHECK_ERR();
}

void nbodyTimer(void * kernelInfo)
{
  KernelInfo * kinfo = reinterpret_cast<KernelInfo * >(kernelInfo);
  Body * storage = new Body[kinfo->n];

  if (dcgn::getRank() == 0)
  {
    int dummy[2];
    FILE * fp = fopen(kinfo->input, "rb");
    fread(dummy, sizeof(dummy), 1, fp);
    fread(storage, sizeof(Body) * kinfo->n, 1, fp);
    fclose(fp);
  }
  dcgn::broadcast(0, storage, sizeof(Body) * kinfo->n);
  dcgn::barrier();
  double t = dcgn::wallTime();
  for (int i = 0; i < kinfo->timeSteps; ++i)
  {
    int sizeIndex = 0, actualSize = dcgn::getSize() * 2 / 3;
    for (int j = 0; j < dcgn::getSize(); ++j)
    {
      if (j % 3 != 0)
      {
        const int start = getIndexForRank(sizeIndex,     actualSize, kinfo->n);
        const int end   = getIndexForRank(sizeIndex + 1, actualSize, kinfo->n);
/*
        for (int k = 0; k < dcgn::globalGPUCount(); ++k)
        {
          int tvar;
          dcgn::recv(dcgn::getGPUID(k, 0), &tvar, sizeof(int), &stat); // printf("rank 1 - end  =%d.\n", tvar); fflush(stdout);
        }
*/
        // dcgn::barrier();
        dcgn::broadcast(j, storage, (end - start) * sizeof(Body));
        ++sizeIndex;
      }
    }
#if BARRIER_BEFORE_BIG_COMM
    dcgn::barrier();
#endif
  }

  dcgn::barrier();
  t = dcgn::wallTime() - t;
  if (dcgn::getRank() == 0)
  {
    printf("done, took %.6f seconds.\n", t);
  }
  delete [] storage;
}

__shared__ char mem[16 * 1024 - 256]; // 16k - 256bytes for some system stuff.
__device__ int buf;

__global__ void nbody(dcgn::GPUInitRequest libParam, Body * gpuBodies, Body * gpuUpdatedBodies, volatile int * sbarr, const int size, const int timeSteps, const float timeDelta)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    // printf("initting.\n"); fflush(stdout);
    dcgn::gpu::init(libParam);
    // printf("%d.%d.%d: broadcasting.\n", dcgn::gpu::getRank(), blockIdx.x, threadIdx.x); fflush(stdout);
    dcgn::gpu::broadcast(0, 0, gpuBodies, size * sizeof(Body));
    // printf("%d.%d.%d: barriering.\n", dcgn::gpu::getRank(), blockIdx.x, threadIdx.x); fflush(stdout);
    dcgn::gpu::barrier(0);
    // printf("%d.%d.%d: executing.\n", dcgn::gpu::getRank(), blockIdx.x, threadIdx.x); fflush(stdout);
  }

  const int START_BODY  = getIndexForRank(dcgn::gpu::getRank(0),      dcgn::gpu::getSize(), size);
  const int END_BODY    = getIndexForRank(dcgn::gpu::getRank(0) + 1,  dcgn::gpu::getSize(), size);

  for (int i = 0; i < timeSteps; ++i)
  {
    updateForces(gpuBodies, gpuUpdatedBodies, START_BODY, END_BODY, size, timeDelta);
    __syncblocks(sbarr);
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
      int sizeIndex = 0, actualSize = dcgn::gpu::getSize() * 2 / 3;
      for (int j = 0; j < dcgn::gpu::getSize(); ++j)
      {
        if (j % 3 != 0)
        {
          const int start = getIndexForRank(sizeIndex,     actualSize, size);
          const int end   = getIndexForRank(sizeIndex + 1, actualSize, size);

          // buf = end;    dcgn::gpu::send(0, 0, &buf, sizeof(int));

          // printf("dcgn::broadcast(slot=%d, root=%d, buf=%p (%p + %d * %d), size=%d( (%d - %d) * %d) )).\n",
          //         0, j, gpuUpdatedBodies + start, gpuUpdatedBodies, start, (int)sizeof(Body), (end - start) * (int)sizeof(Body), end, start, (int)sizeof(Body));
          // fflush(stdout);
          // dcgn::gpu::barrier(0);
          dcgn::gpu::broadcast(0, j, gpuUpdatedBodies + start, (end - start) * sizeof(Body));
          ++sizeIndex;
        }
      }
#if BARRIER_BEFORE_BIG_COMM
      dcgn::gpu::barrier(0);
#endif
    }
    __syncblocks(sbarr);
    Body * tmp = gpuBodies;
    gpuBodies = gpuUpdatedBodies;
    gpuUpdatedBodies = tmp;
  }

  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    dcgn::gpu::barrier(0);
  }
}

__device__ void updateForces(Body * bodies, Body * ubodies, const int startBody, const int endBody, const int n, const float timeDelta)
{
  // load up 2 bodies per thread. then load up one body per thread for work.

  const int BODY_START  = startBody + blockDim.x * blockIdx.x;
  const int BODY_STRIDE = blockDim.x * gridDim.x;

  for (int i = BODY_START; i < endBody; i += BODY_STRIDE)
  {
    const int NUM_BODIES = (i + blockDim.x <= endBody ? blockDim.x : endBody - i);
    Body * sharedBodies = reinterpret_cast<Body * >(mem);
    Body * localBodies  = reinterpret_cast<Body * >(mem + NUM_BODIES * sizeof(Body));
    block_memcpy(sharedBodies, bodies + i, NUM_BODIES * sizeof(Body));
    if (threadIdx.x < NUM_BODIES)
    {
      for (int j = 0; j < n; j += blockDim.x * 2)
      {
        const int NUM_BODIES2 = (j + blockDim.x * 2 <= n ? 2 * blockDim.x : n - j);
        block_memcpy(localBodies, bodies + j, NUM_BODIES2 * sizeof(Body));
        for (int k = 0; k < NUM_BODIES2; ++k)
        {
          sharedBodies[threadIdx.x].addForceFrom(localBodies[k]);
        }
      }
      sharedBodies[threadIdx.x].update(timeDelta);
    }
    block_memcpy2(ubodies + i, sharedBodies, NUM_BODIES * sizeof(Body));
  }
}

__device__ __host__ int getIndexForRank(const int rank, const int size, const int n)
{
  if (rank == 0) return 0;
  if (rank == size) return n;
  float f = static_cast<float>(rank) / static_cast<float>(size) * n;
  return static_cast<int>(f);
}

__device__ void block_memcpy(void * dstMem, const void * srcMem, const int size)
{
        char * dst = reinterpret_cast<       char * >(dstMem);
  const char * src = reinterpret_cast<const  char * >(srcMem);
  int rem = size;

  dst += threadIdx.x * sizeof(float);
  src += threadIdx.x * sizeof(float);

  __syncthreads();
  while (rem >= blockDim.x * sizeof(float))
  {
    *reinterpret_cast<float * >(dst) = *reinterpret_cast<const float * >(src);
    dst += blockDim.x * sizeof(float);
    src += blockDim.x * sizeof(float);
    rem    -= blockDim.x * sizeof(float);
  }
  if (rem > 0 && threadIdx.x == 0)
  {
    while (rem > sizeof(float))
    {
      reinterpret_cast<float * >(dst)[threadIdx.x] = *reinterpret_cast<const float * >(src);
      dst += sizeof(float);
      src += sizeof(float);
      rem    -= sizeof(float);
    }
    while (rem > sizeof(char))
    {
      *(dst++) = *(src++);
      --rem;
    }
  }
  __syncthreads();
}

__device__ void block_memcpy2(void * dstMem, const void * srcMem, const int size)
{
        char * dst = reinterpret_cast<       char * >(dstMem);
  const char * src = reinterpret_cast<const  char * >(srcMem);
  int rem = size;

  dst += threadIdx.x * sizeof(float);
  src += threadIdx.x * sizeof(float);

  __syncthreads();
  while (rem >= blockDim.x * sizeof(float))
  {
    *reinterpret_cast<float * >(dst) = *reinterpret_cast<const float * >(src);
    dst += blockDim.x * sizeof(float);
    src += blockDim.x * sizeof(float);
    rem    -= blockDim.x * sizeof(float);
  }
  if (rem > 0 && threadIdx.x == 0)
  {
    while (rem > sizeof(float))
    {
      reinterpret_cast<float * >(dst)[threadIdx.x] = *reinterpret_cast<const float * >(src);
      dst += sizeof(float);
      src += sizeof(float);
      rem    -= sizeof(float);
    }
    while (rem > sizeof(char))
    {
      *(dst++) = *(src++);
      --rem;
    }
  }
  __syncthreads();
}

__device__ void __syncblocks(volatile int * sbarr)
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    sbarr[blockIdx.x] = 1;
    if (blockIdx.x == 0)
    {
      for (int i = 1; i < gridDim.x; ++i) while (sbarr[i] == 0) { }
      for (int i = 0; i < gridDim.x; ++i) sbarr[i] = 0;
    }
    while (sbarr[blockIdx.x] == 1) { }
  }
  __syncthreads();
}
