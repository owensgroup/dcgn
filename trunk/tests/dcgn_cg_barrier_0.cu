#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>

const int ITERS = 100;

__global__ void kernel(const dcgn::GPUInitRequest libParam)
{
  dcgn::gpu::init(libParam);
  dcgn::gpu::barrier(0);
  for (int i = 0; i < ITERS; ++i)
  {
    dcgn::gpu::barrier(0);
  }
  dcgn::gpu::barrier(0);
}

__host__ void gpuKernel(void * info, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  kernel<<<gridSize, blockSize, sharedMemSize, *stream>>>(libParam);
}

__host__ void gpuDtor(void * info)
{
}

void busySleep()
{
  double t = dcgn::wallTime();
  while (dcgn::wallTime() - t < 0.1) { }
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
  double * t = new double[num];
  for (int i = 0; i < num; ++i) t[i] = vals[i];
  qsort(vals, num, sizeof(double), dcmp);
  for (int i = 0; i < num; ++i)
  {
    printf("%3d: %f   -   %f\n", i, vals[i], t[i]);
    sum += vals[i];
  }
  double mean = sum / num;
  double sigma = 0.0;
  for (int i = 0; i < num; ++i)
  {
    double diff = static_cast<double>(vals[i]) - mean;
    sigma += diff * diff;
  }
  sigma = std::sqrt(sigma / static_cast<double>(num + 1));
  sum = 0.0;
  printf("mean sigma { %f %f }\n", mean, sigma);
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
  delete [] t;
}

void cpuKernel(void * )
{
  double * vals = new double[ITERS];
  int iters = ITERS;
  double t = 0;
  dcgn::barrier();
  for (int i = 0; i < ITERS; ++i)
  {
    if (dcgn::getRank() == 0) busySleep();
    double t2 = dcgn::wallTime();
    dcgn::barrier();
    vals[i] = dcgn::wallTime() - t2;
    t += vals[i];
    if (dcgn::getRank() == 0) busySleep();
  }
  dcgn::barrier();
  if (dcgn::getRank() == 0)
  {
    double sum;
    eliminateOutliers(vals, sum, iters);
    printf(" %d iterations took an average of %.8f sec, adjusted (%d samples) %.8f sec\n", ITERS, (t / ITERS), iters, sum / iters);
    fflush(stdout);
  }
}

int main(int argc, char ** argv)
{
  // int gpus[] = { 0, -1 };
  int gpus[] = { 0, 1, -1 };
  uint3 gs = { 1, 1, 1 }, bs = { 1, 1, 1 };

  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  // dcgn::initCPU(1);
  dcgn::initCPU(2);
  dcgn::initGPU(gpus, 1, 0);
  dcgn::start();

  dcgn::launchCPUKernel(0, cpuKernel, 0);
  dcgn::launchCPUKernel(1, cpuKernel, 0);
  dcgn::launchGPUKernel(0, gpuKernel, gpuDtor,  0, gs, bs);
  dcgn::launchGPUKernel(1, gpuKernel, 0,        0, gs, bs);

  dcgn::finalize();
  return 0;
}
