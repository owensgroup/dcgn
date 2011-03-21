#include <cuda.h>
#include <cuda_runtime_api.h>
#include "samples/Body.cxx"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sys/time.h>

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

double getTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1000000.0;
}

__shared__ char mem[16 * 1024 - 256]; // 16k - 256bytes for some system stuff.

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

__global__ void kernel(Body * bodies, Body * ubodies, const int n, const float timeDelta)
{
  // load up 2 bodies per thread. then load up one body per thread for work.

  const int BODY_START  = blockDim.x * blockIdx.x;
  const int BODY_STRIDE = blockDim.x * gridDim.x;

  for (int i = BODY_START; i < n; i += BODY_STRIDE)
  {
    const int NUM_BODIES = (i + blockDim.x <= n ? blockDim.x : n - i);
    Body * sharedBodies = reinterpret_cast<Body * >(mem);
    Body * localBodies  = reinterpret_cast<Body * >(mem + NUM_BODIES * sizeof(Body));
    block_memcpy(sharedBodies, bodies + i, NUM_BODIES * sizeof(Body));
    if (threadIdx.x < NUM_BODIES)
    {
      for (int j = 0; j < n; j += blockDim.x * 2)
      {
        const int NUM_BODIES2 = (j + blockDim.x * 2 <= n ? 2 * blockDim.x : n - i);
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

__host__ void runKernel(Body * bodies, Body * gpuBodies, Body * gpuUpdatedBodies, const int n, const float timeDelta)
{
  uint3 gs = {  16, 1, 1 };
  uint3 bs = { 128, 1, 1 };
  cudaMemcpy(gpuBodies, bodies, sizeof(Body) * n, cudaMemcpyHostToDevice);        CHECK_ERR();
  kernel<<<gs, bs>>>(gpuBodies, gpuUpdatedBodies, n, timeDelta);                  CHECK_ERR();
  cudaThreadSynchronize();                                                        CHECK_ERR();
  cudaMemcpy(bodies, gpuUpdatedBodies, sizeof(Body) * n, cudaMemcpyDeviceToHost); CHECK_ERR();
}

int main(int argc, char ** argv)
{
  Body * bodies, * gpuBodies, * gpuUpdatedBodies;
  int size, timeSteps;
  double timeDelta;
  FILE * fp, * outfp;

  if (argc != 5)
  {
    fprintf(stderr, "Usage: %s <input_file> <output_file> <time_steps> <time_delta>\n", argv[0]);
    return 1;
  }
  fp = fopen(argv[1], "rb");
  if (!fp)
  {
    fprintf(stderr, "Couldn't open %s for reading\n", argv[1]);
    return 1;
  }
  outfp = fopen(argv[2], "wb");
  if (!outfp)
  {
    fclose(fp);
    fprintf(stderr, "Couldn't open %s for writing\n", argv[2]);
    return 1;
  }

  sscanf(argv[3], "%d", &timeSteps);
  sscanf(argv[4], "%lf", &timeDelta);

  void * debugInfo;
  fread(&size, sizeof(int), 1, fp);
  int dummy; fread(&dummy, sizeof(int), 1, fp);
  cudaMallocHost((void ** )&bodies, size * sizeof(Body));         CHECK_ERR();
  cudaMalloc((void ** )&gpuBodies, size * sizeof(Body));          CHECK_ERR();
  cudaMalloc((void ** )&gpuUpdatedBodies, size * sizeof(Body));   CHECK_ERR();
  cudaMalloc((void ** )&debugInfo, 1048576);                      CHECK_ERR();
  cudaMemset(debugInfo, 0, 1048576);                              CHECK_ERR();
  fread(bodies, sizeof(Body) * size, 1, fp);
  fclose(fp);

  fwrite(&size,       sizeof(int), 1, outfp);
  fwrite(&timeSteps,  sizeof(int), 1, outfp);

  double start = getTime();

  for (int i = 0; i < timeSteps; ++i)
  {
    runKernel(bodies, gpuBodies, gpuUpdatedBodies, size, timeDelta);
    fwrite(bodies, sizeof(Body) * size, 1, outfp);
  }
  fclose(outfp);

  double time = getTime() - start;

  printf("done, took %.3fs.\n", time);

  cudaFreeHost(bodies);       CHECK_ERR();
  cudaFree(gpuBodies);        CHECK_ERR();
  cudaFree(gpuUpdatedBodies); CHECK_ERR();
  cudaFree(debugInfo);        CHECK_ERR();

  return 0;
}
