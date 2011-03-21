#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>

/*
const int NUM       = 17;
const int MIN_SIZE  = 1;
const int MAX_SIZE  = 1024 << (NUM - 1);
const int ITERS     = 30;
*/

const int NUM       = 20;
const int MAX_SIZE  = 1 << (NUM - 1);
const int ITERS     = 30;

__shared__ clock_t timers[NUM];

struct Mem
{
  void * gmem;
  clock_t * clocks;
  clock_t * startClocks;
  clock_t * stopClocks;
  int     * iters;
  int       i;
  bool      primary;
};

__global__ void kernel(void     * gmem,
                       clock_t  * clocks,
                       clock_t  * startClocks,
                       clock_t  * stopClocks,
                       int      * iters,
                       int        i,
                       const dcgn::GPUInitRequest libParam)
{
  dcgn::gpu::init(libParam);
  int size = 1 << i;
  *clocks = 0;
  dcgn::gpu::barrier(0);
  int iterations = 0;
  for (int j = 0; j < ITERS; ++j)
  {
    clock_t c0 = clock();
    if (dcgn::gpu::getRank(0) == 0)
    {
      while (clock() > c0 && clock() - c0 < 1000000) { }
    }
    clock_t c1 = clock();
    dcgn::gpu::broadcast(0, 0, gmem, size);
    clock_t c2 = clock();
    if (c1 < c2)
    {
      ++iterations;
      *clocks += c2 - c1;
    }
    if (dcgn::gpu::getRank(0) == 0)
    {
      while (clock() > c2 && clock() - c2 < 1000000) { }
    }
    *(startClocks++) = c1;
    *(stopClocks ++) = c2;
  }
  *iters = iterations;
  dcgn::gpu::barrier(0);
}

__host__ void gpuKernel(void * info, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  Mem * mem = reinterpret_cast<Mem * >(info);

  fprintf(stderr, "%d:1 - mem: %p\n  gmem: %p\n  clocks: %p\n  start: %p\n  stop: %p\n  iters: %p\n",
                  static_cast<int>(dcgn::getRank()), mem, mem->gmem, mem->clocks, mem->startClocks, mem->stopClocks, mem->iters);
  fflush(stderr);

  cudaMalloc(reinterpret_cast<void ** >(&mem->gmem),        MAX_SIZE * 2);
  cudaMalloc(reinterpret_cast<void ** >(&mem->clocks),      sizeof(clock_t));
  cudaMalloc(reinterpret_cast<void ** >(&mem->startClocks), sizeof(clock_t) * ITERS);
  cudaMalloc(reinterpret_cast<void ** >(&mem->stopClocks),  sizeof(clock_t) * ITERS);
  cudaMalloc(reinterpret_cast<void ** >(&mem->iters),       sizeof(int));

  fprintf(stderr, "%d:2 - mem: %p\n  gmem: %p\n  clocks: %p\n  start: %p\n  stop: %p\n  iters: %p\n",
                  static_cast<int>(dcgn::getRank()), mem, mem->gmem, mem->clocks, mem->startClocks, mem->stopClocks, mem->iters);
  fflush(stderr);

  kernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(mem->gmem,
                                                          mem->clocks, mem->startClocks, mem->stopClocks, mem->iters, mem->i,
                                                          libParam);
}

static inline int clockCmp(const void * a, const void * b)
{
  const clock_t & c1 = *reinterpret_cast<const clock_t * >(a);
  const clock_t & c2 = *reinterpret_cast<const clock_t * >(b);
  if (c1 < c2) return -1;
  if (c1 > c2) return 1;
  return 0;
}

void eliminateOutliers(clock_t * startClocks, clock_t * stopClocks, int & numClocks)
{
  double sum = 0.0;
  for (int i = 0; i < numClocks; )
  {
    unsigned int c1 = startClocks[i] & 0x7FFFFFFF;
    unsigned int c2 = stopClocks [i] & 0x7FFFFFFF;
    if (c1 < c2)
    {
      ++i;
      startClocks[i] = c1;
      stopClocks [i] = c2;
      sum = static_cast<double>(c2 - c1);
    }
    else
    {
      memmove(startClocks + i, startClocks + i + 1, sizeof(clock_t) * numClocks - i - 1);
      memmove(stopClocks  + i, stopClocks  + i + 1, sizeof(clock_t) * numClocks - i - 1);
      --numClocks;
    }
  }
  double mean = sum / numClocks;
  double sigma = 0.0;
  for (int i = 0; i < numClocks; ++i)
  {
    double diff = static_cast<double>(stopClocks[i] - startClocks[i]) - mean;
    sigma += diff * diff;
  }
  sigma /= numClocks + 1;
  sigma = std::sqrt(sigma);
  for(int i = 0; i < numClocks; )
  {
    double diff = stopClocks[i] - startClocks[i];
    double dev  = std::abs(mean - diff);
    if (dev / sigma > 2.0)
    {
      memmove(startClocks + i, startClocks + i + 1, sizeof(clock_t) * numClocks - i - 1);
      memmove(stopClocks  + i, stopClocks  + i + 1, sizeof(clock_t) * numClocks - i - 1);
      --numClocks;
    }
    else
    {
      ++i;
    }
  }
}

__host__ void gpuDtor(void * info)
{
  Mem * mem = reinterpret_cast<Mem * >(info);

  if (dcgn::getRank() == 0)
  {
    clock_t   clocks;
    clock_t * startClocks = new clock_t[ITERS];
    clock_t * stopClocks  = new clock_t[ITERS];
    clock_t * diffClocks  = new clock_t[ITERS];
    int       iters;

    static CUdevice dev;
    static CUdevprop prop;
    if (mem->i == 0)
    {
      cuInit(0);
      cuDeviceGet(&dev, 0);
      cuDeviceGetProperties(&prop, dev);
    }

    cudaMemcpy(&clocks,     mem->clocks,      sizeof(clock_t),          cudaMemcpyDeviceToHost);
    cudaMemcpy(startClocks, mem->startClocks, sizeof(clock_t) * ITERS,  cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClocks,  mem->stopClocks,  sizeof(clock_t) * ITERS,  cudaMemcpyDeviceToHost);
    cudaMemcpy(&iters,      mem->iters,       sizeof(int),              cudaMemcpyDeviceToHost);

    unsigned int sum1 = 0;
    iters = ITERS;
    eliminateOutliers(startClocks, stopClocks, iters);
    for (int j = 0; j < ITERS; ++j)
    {
      unsigned int c1 = startClocks[j];
      unsigned int c2 = stopClocks [j];
      if (c2 > c1)
      {
        sum1 += c2 / c1;
        ++iters;
        diffClocks[j] = c2 - c1;
      }
    }
    qsort(diffClocks, iters, sizeof(clock_t), clockCmp);
    printf("%3d %s: %20.6f\n", 1 << (mem->i - 10 * (mem->i / 10)),
                               mem->i < 10 ? " B" : mem->i < 20 ? "kB" : "MB",
                               // (((double)sum1 / ITERS) / (double)prop.clockRate) / 1000.0);
                               (((double)diffClocks[iters / 2]) / (double)prop.clockRate) / 1000.0);
    fflush(stdout);
  }

  fprintf(stderr, "%d:3 - mem: %p\n  gmem: %p\n  clocks: %p\n  start: %p\n  stop: %p\n  iters: %p\n",
                  static_cast<int>(dcgn::getRank()), mem, mem->gmem, mem->clocks, mem->startClocks, mem->stopClocks, mem->iters);
  fflush(stderr);

  // cudaFree(mem->gmem);
  // cudaFree(mem->clocks);
  // cudaFree(mem->startClocks);
  // cudaFree(mem->stopClocks);
  // cudaFree(mem->iters);
  // delete mem;
}

int main(int argc, char ** argv, char ** envp)
{
  int gpus[] = { 0, 1, -1 };
  uint3 gs = { 1, 1, 1 }, bs = { 1, 1, 1 };

  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  dcgn::initCPU(0);
  dcgn::initGPU(gpus, 1, 0);
  dcgn::start();

  for (int i = 0; i < NUM; ++i)
  {
    Mem * gpuMem1 = new Mem;
    Mem * gpuMem2 = new Mem;
    fprintf(stderr, "0:0 - mem: %p\n", gpuMem1);
    fprintf(stderr, "0:1 - mem: %p\n", gpuMem1);
    fflush(stderr);
    gpuMem1->i = i;
    gpuMem2->i = i;
    gpuMem1->primary = true;
    gpuMem2->primary = false;
    dcgn::launchGPUKernel(0, gpuKernel, gpuDtor,  gpuMem1, gs, bs);
    dcgn::launchGPUKernel(1, gpuKernel, gpuDtor,  gpuMem2, gs, bs);
  }

  dcgn::finalize();
  return 0;
}
