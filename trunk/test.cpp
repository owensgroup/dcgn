#include <sys/time.h>
#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>

double getTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}
double stddev(const double * const arr, const int n)
{
  double avg = 0.0, sum = 0.0, sq = 0.0;
  for (int i = 0; i < n; ++i) 
  {
    sq+= arr[i] * arr[i];
    avg += arr[i];
  }
  avg /= (double)n;
  return std::sqrt((sq - n * avg * avg) / (double)n);
}

int main(int argc, char * argv)
{
  const int NUM_ITERS = 400;
  double times[NUM_ITERS], tt = 0.0;
  void * t0, * t1;
  cudaMalloc(&t0, sizeof(float4));
  cudaMallocHost(&t1, sizeof(float4));

  printf("test     | Minimum        | Mean           | Median         | Maximum        | Std. Dev       |\n");
  tt = 0.0;
  for (int i = 0; i < NUM_ITERS; ++i)
  {
    times[i] = getTime();
    cudaMemset(t0, 0, sizeof(float4));
    times[i] = getTime() - times[i];
    tt += times[i];
  }
  std::sort(times, times + NUM_ITERS);
  printf("memset   | %14.10f | %14.10f | %14.10f | %14.10f | %14.10f |\n", times[0], tt / NUM_ITERS, times[NUM_ITERS / 2], times[NUM_ITERS - 1], stddev(times, NUM_ITERS));
  
  tt = 0.0;
  for (int i = 0; i < NUM_ITERS; ++i)
  {
    times[i] = getTime();
    cudaMemcpy(t0, t1, sizeof(float4), cudaMemcpyHostToDevice);
    times[i] = getTime() - times[i];
    tt += times[i];
  }
  std::sort(times, times + NUM_ITERS);
  printf("cpu->gpu | %14.10f | %14.10f | %14.10f | %14.10f | %14.10f |\n", times[0], tt / NUM_ITERS, times[NUM_ITERS / 2], times[NUM_ITERS - 1], stddev(times, NUM_ITERS));

  tt = 0.0;
  for (int i = 0; i < NUM_ITERS; ++i)
  {
    times[i] = getTime();
    cudaMemcpy(t1, t0, sizeof(float4), cudaMemcpyDeviceToHost);
    times[i] = getTime() - times[i];
    tt += times[i];
  }
  std::sort(times, times + NUM_ITERS);
  printf("gpu->cpu | %14.10f | %14.10f | %14.10f | %14.10f | %14.10f |\n", times[0], tt / NUM_ITERS, times[NUM_ITERS / 2], times[NUM_ITERS - 1], stddev(times, NUM_ITERS));

  cudaFree(t0);
  cudaFreeHost(t1);
  return 0;
}

