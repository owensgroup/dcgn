#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>

const int ITERS = 100;

__shared__ clock_t timers[10];

struct Mem { clock_t * timer; int * iters; clock_t * starts, * stops; };

__global__ void kernel(clock_t * gpuClock, int * iters, clock_t * starts, clock_t * stops, const dcgn::GPUInitRequest libParam)
{
  dcgn::gpu::init(libParam);
  dcgn::gpu::barrier(0);
  if (dcgn::gpu::getRank(0) == 0)
  {
    *gpuClock = 0;
    *iters = 0;
  }
  for (int i = 0; i < ITERS; ++i)
  {
    clock_t c0 = clock();
    if (dcgn::gpu::getRank(0) == 0)
    {
      while (clock() > c0 && clock() - c0 < 1000000) { }
    }
    __syncthreads();
    timers[0] = clock();
    dcgn::gpu::barrier(0);
    timers[1] = clock();
    __syncthreads();
    if (dcgn::gpu::getRank(0) == 0)
    {
      *(starts++) = timers[0];
      *(stops++)  = timers[1];
      *gpuClock += timers[1] - timers[0];
      ++*iters;
    }
  }
  dcgn::gpu::barrier(0);
}

int dcmp(const void * a, const void * b)
{
  const double & d1 = *(const double * )a;
  const double & d2 = *(const double * )b;
  if (d1 < d2) return -1;
  if (d1 > d2) return  1;
  return 0;
}

void eliminateOutliers(double * vals, double & sum, int & num)
{
  sum = 0.0;
  qsort(vals, num, sizeof(double), dcmp);
  for (int i = 0; i < num; ++i)
  {
    printf("%3d: %f\n", i, vals[i]);
    sum += vals[i];
  }
  double mean = sum / num;
  double sigma = 0.0;
  for (int i = 0; i < num; ++i)
  {
    double diff = static_cast<double>(vals[i]) - mean;
    sigma += diff * diff;
  }
  sigma /= num + 1;
  sigma = std::sqrt(sigma);
  qsort(vals, num, sizeof(double), dcmp);
  printf("mean sigma { %f %f }\n", mean, sigma);
  sum = 0.0;
  for(int i = 0; i < num; )
  {
    double dev  = std::abs(mean - vals[i]);
    if (dev / sigma > 2.0)
    {
      memmove(vals + i, vals + i + 1, sizeof(double) * (num - i - 1));
      --num;
    }
    else
    {
      sum += vals[i];
      ++i;
    }
  }
}

__host__ void gpuKernel(void * info, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  Mem * mem = (Mem * )info;
  cudaMalloc((void ** )&mem->timer,  sizeof(clock_t));
  cudaMalloc((void ** )&mem->iters,  sizeof(int));
  cudaMalloc((void ** )&mem->starts, sizeof(clock_t) * ITERS);
  cudaMalloc((void ** )&mem->stops,  sizeof(clock_t) * ITERS);
  kernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(mem->timer, mem->iters, mem->starts, mem->stops, libParam);
}

__host__ void gpuDtor(void * info)
{
  if (dcgn::getNodeID() != 0) return;
  Mem * ptr;
  int iters;
  clock_t gpuClock;
  clock_t * starts, * stops;
  double * diffs = new double[ITERS];

  int deviceIndex = 0;
  CUdevice dev;
  CUdevprop prop;
  cuInit(0);
  cuDeviceGet(&dev, deviceIndex);
  cuDeviceGetProperties(&prop, dev);

  starts = new clock_t[ITERS];
  stops = new clock_t[ITERS];
  ptr = (Mem * )info;
  cudaMemcpy(&gpuClock, ptr->timer,  sizeof(clock_t),         cudaMemcpyDeviceToHost);
  cudaMemcpy(&iters,    ptr->iters,  sizeof(int),             cudaMemcpyDeviceToHost);
  cudaMemcpy(starts,    ptr->starts, sizeof(clock_t) * ITERS, cudaMemcpyDeviceToHost);
  cudaMemcpy(stops,     ptr->stops,  sizeof(clock_t) * ITERS, cudaMemcpyDeviceToHost);

  // clock_t sum = 0;
  // clock_t sum2 = 0;
  // clock_t sum3 = 0;
  double t = 0.0;
  iters = 0;
  for (int i = 0; i < ITERS; ++i)
  {
    unsigned int c1 = stops[i]  & 0x7FFFFFFF;
    unsigned int c2 = starts[i] & 0x7FFFFFFF;
    if (c1 > c2)
    {
      diffs[iters] = static_cast<double>(c1 - c2) / static_cast<double>(prop.clockRate) / 1000.0;
      t += diffs[iters];
      ++iters;
    }
    // printf("%20u - %20u = %20u\n", c1, c2, c1 - c2);
    // sum += c1 - c2;
    // sum2 += stops[i] - starts[i];
    // // diffs[i] = c1 - c2;
    // if (stops[i] > starts[i])
    // {
    //   sum3 += stops[i] - starts[i];
    // }
  }

  // printf("%d iterations took an average of %.8f sec\n", iters, ((double)sum       / ITERS / (double)prop.clockRate) / 1000.0);
  // printf("%d iterations took an average of %.8f sec\n", iters, ((double)sum2      / ITERS / (double)prop.clockRate) / 1000.0);
  // printf("%d iterations took an average of %.8f sec\n", iters, ((double)sum3      / iters / (double)prop.clockRate) / 1000.0);
  // printf("%d iterations took an average of %.8f sec\n", iters, ((double)gpuClock  / iters / (double)prop.clockRate) / 1000.0);
  double dsum = 0.0;
  int origIters = iters;
  eliminateOutliers(diffs, dsum, iters);
  printf(" %d iterations took an average of %.8f sec, adjusted (%d samples) %.8f sec\n", origIters, (t / origIters), iters, dsum / iters);
  fflush(stdout);
  cudaFree(ptr->timer);
  cudaFree(ptr->iters);
  cudaFree(ptr->starts);
  cudaFree(ptr->stops);
}

int main(int argc, char ** argv)
{
  Mem gpuMem;
  int gpus[] = { 0, 1, -1 };
  uint3 gs = { 1, 1, 1 }, bs = { 1, 1, 1 };

  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  dcgn::initCPU(0);
  dcgn::initGPU(gpus, 1, 0);
  dcgn::start();

  dcgn::launchGPUKernel(0, gpuKernel, gpuDtor,  &gpuMem, gs, bs);
  dcgn::launchGPUKernel(1, gpuKernel, 0,        &gpuMem, gs, bs);

  dcgn::finalize();
  return 0;
}
