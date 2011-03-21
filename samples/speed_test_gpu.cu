#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <algorithm>

typedef struct _KernelInfo
{
  int gpuID;
  int id, size;
  int testNum, unitNum;
  double frequency;
  void * gpuBuf;
  uint3 gs, bs;
  clock_t * gpuClocks;
  clock_t * cpuClocks;
} KernelInfo;

enum
{
  TEST_LOCAL_SENDRECV,
  TEST_GLOBAL_SEND,
  TEST_GLOBAL_RECV,
  TEST_GLOBAL_SENDRECV,
  TEST_GLOBAL_BARRIER,
  TEST_GLOBAL_BROADCAST1,
  TEST_GLOBAL_BROADCAST2,
  TEST_GLOBAL_BROADCAST3,
  TEST_GLOBAL_BROADCAST4,
  TEST_GLOBAL_BROADCAST5,
};

const int NUM_TESTS = 8;

const int NUM_UNITS = 17;
const int NUM_ITERS = 30;

__host__    void timer(void * param, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream);
            void timerCleanup(void * param);
__global__  void speedTestKernel(dcgn::GPUInitRequest param, void * buf, clock_t * clocks, const int testNum, const int unitNum);

const char * const TEST_NAMES[] =
{
  "Local SendRecvReplace",
  "Global Send",
  "Global Recv",
  "Global SendRecvReplace",
  "Global Barrier",
  "Global Broadcast Send",
  "Global Broadcast Recv",
  "Global Broadcast Recv 2",
  "Global Broadcast Recv 3",
  "Global Broadcast Recv 4",
};

int main(int argc, char ** argv)
{
  uint3 gs = { 1, 1, 1 }, bs = { 1, 1, 1 };
  int gpus[] = { 0, 1, -1 };
  dcgn::initAll(&argc, &argv, 0, gpus, 1, 0, 0);

  KernelInfo kernelInfo1, kernelInfo2;
  kernelInfo1.frequency = kernelInfo2.frequency = -1.0;
  kernelInfo1.unitNum   = kernelInfo2.unitNum = 0;
  kernelInfo1.testNum   = kernelInfo2.testNum = 0;
  kernelInfo1.bs        = kernelInfo2.bs = bs;
  kernelInfo1.gs        = kernelInfo2.gs = gs;
  kernelInfo1.gpuBuf    = kernelInfo2.gpuBuf = 0;
  kernelInfo1.gpuID = 0;
  kernelInfo2.gpuID = 1;

  dcgn::launchGPUKernel(0, timer, timerCleanup, &kernelInfo1, gs, bs);
  dcgn::launchGPUKernel(1, timer, timerCleanup, &kernelInfo2, gs, bs);

  while (!dcgn::areAllLocalResourcesIdle())
  {
    sched_yield();
  }

  dcgn::finalize();

  return 0;
}

int dcmp(const void * a, const void * b)
{
  const double & d0 = *reinterpret_cast<const double * >(a);
  const double & d1 = *reinterpret_cast<const double * >(b);
  if (d0 < d1) return -1;
  if (d0 > d1) return 1;
  return 0;
}

double average(const double * const arr, const int numIters)
{
  double ret = 0.0;
  for (int i = 0; i < numIters; ++i) ret += arr[i];
  return ret / static_cast<double>(numIters);
}

double stddev(const double * const arr, const int numIters)
{
  double avg = 0.0, sq = 0.0;
  for (int i = 0; i < numIters; ++i)
  {
    sq += arr[i] * arr[i];
    avg += arr[i];
  }
  avg /= static_cast<double>(numIters);
  return std::sqrt((sq - static_cast<double>(numIters) * avg * avg) / static_cast<double>(numIters));
}

void timer(void * param, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  KernelInfo * kinfo = reinterpret_cast<KernelInfo * >(param);

  if (kinfo->gpuBuf == 0)
  {
    cudaMalloc    (                           &kinfo->gpuBuf,     256 * 1024 * 1024);
    cudaMalloc    (reinterpret_cast<void ** >(&kinfo->gpuClocks), sizeof(clock_t) * NUM_ITERS * 2);
    cudaMallocHost(reinterpret_cast<void ** >(&kinfo->cpuClocks), sizeof(clock_t) * NUM_ITERS * 2);
    cudaMemset    (kinfo->gpuClocks, 0, sizeof(clock_t) * NUM_ITERS * 2);
    kinfo->id   = libParam.gpuRank;
    kinfo->size = libParam.gpuSize;
  }

  speedTestKernel<<<gridSize, blockSize, 0, *stream>>>(libParam, kinfo->gpuBuf, kinfo->gpuClocks, kinfo->testNum, kinfo->unitNum);
}
void timerCleanup(void * param)
{
  double runTimes[NUM_ITERS];
  KernelInfo * kinfo = reinterpret_cast<KernelInfo * >(param);
  clock_t * clocks = kinfo->cpuClocks;

  cudaMemcpy(kinfo->cpuClocks, kinfo->gpuClocks, sizeof(clock_t) * NUM_ITERS * 2, cudaMemcpyDeviceToHost);
  if (kinfo->frequency == -1.0)
  {
    CUdevice dev;
    CUdevprop prop;
    cuInit(0);
    cuDeviceGet(&dev, kinfo->id % 2);
    cuDeviceGetProperties(&prop, dev);
    kinfo->frequency = static_cast<double>(prop.clockRate) * 1000.0;
  }

  if (kinfo->id == 0)
  {
    int i = kinfo->testNum;
    int j = kinfo->unitNum;
    if (kinfo->unitNum == 0)
    {
      printf("%-25s - %4d CPUs.\n", TEST_NAMES[i], kinfo->size);
      printf("           | Min. Time   | Max. Time   | Mean Time   | Median Time |\n");
      printf("+----------+-------------+-------------+-------------+-------------+\n");
    }
    for (int k = 0; k < NUM_ITERS; ++k)
    {
      unsigned long long int c0 = static_cast<unsigned long long int>(*(clocks++));
      unsigned long long int c1 = static_cast<unsigned long long int>(*(clocks++));
      if (c1 < c0)
      {
        c0 -= c1 + 1;
        c1 = (unsigned long long int)-1;
        // printf("\n(%llu-%llu)=%llu=%f/%llu=%f\n\n", c1, c0, c1 - c0, static_cast<double>(c1 - c0), static_cast<unsigned long long int>(frequency), static_cast<double>(c1 - c0) / frequency);
      }
      runTimes[k] = static_cast<double>(c1 - c0) / kinfo->frequency;
      if (k > 0 && c1 - c0 > kinfo->frequency * 1000)
      {
        runTimes[k] = runTimes[k - 1];
      }
    }
    qsort(runTimes, NUM_ITERS, sizeof(double), dcmp);
    char units[40];
    if (i == TEST_GLOBAL_BARRIER)
    {
      strcpy(units, "100");
    }
    else
    {
      const int amt = 1024 << j;
      if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
      else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
    }
    double avg = average(runTimes, NUM_ITERS);
    double sd = stddev(runTimes, NUM_ITERS);
    int lowIndex = 0, hiIndex = NUM_ITERS - 1;
    while ((runTimes[lowIndex] < avg - 2 * sd) && lowIndex < NUM_ITERS) ++lowIndex;
    while ((runTimes[hiIndex ] > avg + 2 * sd) && hiIndex  > lowIndex)  --hiIndex;
    avg = average(runTimes + lowIndex, hiIndex - lowIndex + 1);
    printf("| %-8s | %11f | %11f | %11f | %11f | %d\n", units, runTimes[lowIndex], runTimes[hiIndex], avg, runTimes[lowIndex + (hiIndex - lowIndex) / 2], hiIndex - lowIndex + 1);
    if (kinfo->unitNum == NUM_UNITS - 1)
    {
      printf("+----------+-------------+-------------+-------------+-------------+\n");
    }
    fflush(stdout);
  }

  if (kinfo->testNum == NUM_TESTS - 1 && kinfo->unitNum == NUM_UNITS - 1)
  {
    cudaFree    (kinfo->gpuClocks);
    cudaFree    (kinfo->gpuBuf);
    cudaFreeHost(kinfo->cpuClocks);
  }
  else
  {
    kinfo->unitNum++;
    if (kinfo->unitNum == NUM_UNITS)
    {
      kinfo->unitNum = 0;
      kinfo->testNum++;
    }
    dcgn::launchGPUKernel(kinfo->gpuID, timer, timerCleanup, param, kinfo->gs, kinfo->bs);
  }
}

__device__ void localSendRecvTest       (const int param, void * buf, clock_t * clocks);
__device__ void globalSendTest          (const int param, void * buf, clock_t * clocks);
__device__ void globalRecvTest          (const int param, void * buf, clock_t * clocks);
__device__ void globalSendRecvTest      (const int param, void * buf, clock_t * clocks);
__device__ void globalBarrierTest       (const int param, void * buf, clock_t * clocks);
__device__ void globalBroadcastSendTest (const int param, void * buf, clock_t * clocks);
__device__ void globalBroadcastRecvTest (const int param, void * buf, clock_t * clocks);
__device__ void globalBroadcastRecvTest2(const int param, void * buf, clock_t * clocks);
__device__ void globalBroadcastRecvTest3(const int param, void * buf, clock_t * clocks);
__device__ void globalBroadcastRecvTest4(const int param, void * buf, clock_t * clocks);

typedef void (*SpeedTestFunc)(const int param, void * buf, clock_t * clocks);

__global__  void speedTestKernel(dcgn::GPUInitRequest param, void * buf, clock_t * clocks, const int testNum, const int unitNum)
{
  dcgn::gpu::init(param);

#define RUN_TEST(test)                  \
  for (int j = 0; j < NUM_ITERS; ++j)   \
  {                                     \
    test(unitNum, buf, clocks);         \
    clocks += 2;                        \
  }                                     \

  dcgn::gpu::barrier(0);
  switch (testNum)
  {
  case TEST_LOCAL_SENDRECV:     RUN_TEST(localSendRecvTest       ); break;
  case TEST_GLOBAL_SEND:        RUN_TEST(globalSendTest          ); break;
  case TEST_GLOBAL_RECV:        RUN_TEST(globalRecvTest          ); break;
  case TEST_GLOBAL_SENDRECV:    RUN_TEST(globalSendRecvTest      ); break;
  case TEST_GLOBAL_BARRIER:     RUN_TEST(globalBarrierTest       ); break;
  case TEST_GLOBAL_BROADCAST1:  RUN_TEST(globalBroadcastSendTest ); break;
  case TEST_GLOBAL_BROADCAST2:  RUN_TEST(globalBroadcastRecvTest ); break;
  case TEST_GLOBAL_BROADCAST3:  RUN_TEST(globalBroadcastRecvTest2); break;
  case TEST_GLOBAL_BROADCAST4:  RUN_TEST(globalBroadcastRecvTest3); break;
  case TEST_GLOBAL_BROADCAST5:  RUN_TEST(globalBroadcastRecvTest4); break;
  }
  dcgn::gpu::barrier(0);
}

__device__ void localSendRecvTest(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;
  if (dcgn::gpu::getRank(0) == 0)
  {
    dcgn::CommStatus stat;
    *(clocks++) = clock();
    dcgn::gpu::sendRecvReplace(0, dcgn::gpu::getRank(0), dcgn::gpu::getRank(0), buf, amt, &stat);
    *(clocks++) = clock();
  }
}

__device__ void globalSendTest(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;
  if (dcgn::gpu::getRank(0) == 0)
  {
    *(clocks++) = clock();
    dcgn::gpu::send(0, 2, buf, amt);
    *(clocks++) = clock();
  }
  else if (dcgn::gpu::getRank(0) == 2)
  {
    dcgn::CommStatus stat;
    *(clocks++) = clock();
    dcgn::gpu::recv(0, 0, buf, amt, &stat);
    *(clocks++) = clock();
  }
}

__device__ void globalRecvTest(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;
  if (dcgn::gpu::getRank(0) == 2)
  {
    *(clocks++) = clock();
    dcgn::gpu::send(0, 0, buf, amt);
    *(clocks++) = clock();
  }
  else if (dcgn::gpu::getRank(0) == 0)
  {
    dcgn::CommStatus stat;
    *(clocks++) = clock();
    dcgn::gpu::recv(0, 2, buf, amt, &stat);
    *(clocks++) = clock();
  }
}

__device__ void globalSendRecvTest(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;
  if (dcgn::gpu::getRank(0) == 0 || dcgn::gpu::getRank(0) == 2)
  {
    dcgn::CommStatus stat;
    *(clocks++) = clock();
    dcgn::gpu::sendRecvReplace(0, 2 - dcgn::gpu::getRank(0), 2 - dcgn::gpu::getRank(0), buf, amt, &stat);
    *(clocks++) = clock();
  }
}

__device__ void globalBarrierTest(const int param, void * buf, clock_t * clocks)
{
  *(clocks++) = clock();
  for (int i = 0; i < 100; ++i)
  {
    dcgn::gpu::barrier(0);
  }
  *(clocks++) = clock();
}

__device__ void globalBroadcastSendTest(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;

  *(clocks++) = clock();
  dcgn::gpu::broadcast(0, 0, buf, amt);
  *(clocks++) = clock();
}

__device__ void globalBroadcastRecvTest(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;

  *(clocks++) = clock();
  dcgn::gpu::broadcast(0, 1, buf, amt);
  *(clocks++) = clock();
}

__device__ void globalBroadcastRecvTest2(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;

  *(clocks++) = clock();
  dcgn::gpu::broadcast(0, 2, buf, amt);
  *(clocks++) = clock();
}

__device__ void globalBroadcastRecvTest3(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;

  *(clocks++) = clock();
  dcgn::gpu::broadcast(0, 3, buf, amt);
  *(clocks++) = clock();
}

__device__ void globalBroadcastRecvTest4(const int param, void * buf, clock_t * clocks)
{
  const int amt = 1024 << param;

  *(clocks++) = clock();
  dcgn::gpu::broadcast(0, 6, buf, amt);
  *(clocks++) = clock();
}
