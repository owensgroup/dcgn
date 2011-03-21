#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <sm_11_atomic_functions.h>
#include <cstdio>

typedef struct _MandelbrotInfo
{
  int width, height, maxRows, maxIters;
  float xMin, xMax, yMin, yMax;
} MandelbrotInfo;
typedef struct _CommInfo
{
  int terminate;
  int startRow, endRow;
} CommInfo;

__device__ MandelbrotInfo mandelbrotInfo;
__device__ CommInfo commInfo;

__device__ float runPixel(const int pixX, const int pixY)
{
  int iter;
  float z, zi, mag;
  float x = static_cast<float>(pixX) / static_cast<float>(mandelbrotInfo.width   - 1);
  float y = static_cast<float>(pixY) / static_cast<float>(mandelbrotInfo.height  - 1);

  z = zi = mag = 0.0f;
  x = mandelbrotInfo.xMin + (mandelbrotInfo.xMax - mandelbrotInfo.xMin) * x;
  y = mandelbrotInfo.yMin + (mandelbrotInfo.yMax - mandelbrotInfo.yMin) * y;

  for (iter = 0; mag < 4.0f && iter < mandelbrotInfo.maxIters; ++iter)
  {
    const float t = z * z - zi * zi + x;
    zi = 2 * z * zi + y;
    z = t;
    mag = z * z + zi * zi;
  }

  return static_cast<float>(iter) / static_cast<float>(mandelbrotInfo.maxIters);
}

__device__ void localBarrier(volatile unsigned int * mutexVals)
{
  __syncthreads();

  if (threadIdx.x == 0)
  {
    mutexVals[blockIdx.x] = 1;
    if (blockIdx.x == 0)
    {
      for (int i = 1; i < gridDim.x; ++i) { while (mutexVals[i] == 0) { } }
      for (int i = 0; i < gridDim.x; ++i) { mutexVals[i] = 0; }
    }
    while (mutexVals[blockIdx.x] == 1) { }
  }

  __syncthreads();
}

__global__ void mandelbrot(float * pixels, volatile unsigned int * mutexVals, const dcgn::GPUInitRequest libParam)
{
  dcgn::gpu::init(libParam);

  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    dcgn::gpu::broadcast(0, 0, &mandelbrotInfo, sizeof(mandelbrotInfo));
  }

  commInfo.terminate = 0;
  while (!commInfo.terminate)
  {
    localBarrier(mutexVals);
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
      dcgn::gpu::recv(0, 0, &commInfo, sizeof(commInfo), 0);
    }
    localBarrier(mutexVals);
    if (commInfo.terminate) break;

    const int NUM_PIXELS = (commInfo.endRow - commInfo.startRow) * mandelbrotInfo.width;
    const int STRIDE = blockDim.x * gridDim.x;
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    float * ptr = pixels;
    float * end = pixels + NUM_PIXELS;

    while (ptr + offset < end)
    {
      *(ptr + offset) = runPixel(offset % mandelbrotInfo.width, commInfo.startRow + offset / mandelbrotInfo.width);
      offset += STRIDE;
    }
    localBarrier(mutexVals);
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
      dcgn::gpu::send(0, 0, &commInfo, sizeof(commInfo));
      dcgn::gpu::send(0, 0, pixels, sizeof(float) * mandelbrotInfo.width * (commInfo.endRow - commInfo.startRow));
    }
  }
  __syncthreads();
}

#define checkError(x)                                                                                                               \
{                                                                                                                                   \
  cudaError_t err = cudaGetLastError();                                                                                             \
  if (err != cudaSuccess)                                                                                                           \
  {                                                                                                                                 \
    fprintf(stderr, "%s - Error on line %s.%s.%d - %s. Aborting.\n", x, __FILE__, __FUNCTION__, __LINE__, cudaGetErrorString(err)); \
    fflush(stderr);                                                                                                                 \
    while (true) *(int *)0 = 0;                                                                                                     \
  }                                                                                                                                 \
}                                                                                                                                   \

__host__ void gpuKernelWrapper(void * minfo, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  float * pixels;
  void * devMutexVals;
  MandelbrotInfo * manInfo = reinterpret_cast<MandelbrotInfo * >(minfo);
  cudaMalloc(reinterpret_cast<void ** >(&devMutexVals), gridSize.x);                                  checkError("malloc");
  cudaMalloc(reinterpret_cast<void ** >(&pixels), sizeof(float) * manInfo->maxRows * manInfo->width); checkError("malloc");
  cudaMemset(pixels, 0, sizeof(float) * manInfo->maxRows * manInfo->width);                           checkError("memset");
  cudaMemset(devMutexVals, 0, sizeof(unsigned int) * gridSize.x);                                     checkError("memset");
  mandelbrot<<<gridSize, blockSize, sharedMemSize, *stream>>>(pixels, reinterpret_cast<volatile unsigned int * >(devMutexVals), libParam);
  checkError("mandelbrot");
}
