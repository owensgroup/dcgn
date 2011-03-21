#include <dcgn/dcgn.h>
#include <algorithm>
#include <cmath>

typedef double (* SpeedTest)(const int id, const int rank, const int param, char * const units);

void timer(void * dummy);

double localSendRecvTest(const int id, const int rank, const int param, char * const units);

double globalSendTest(const int id, const int rank, const int param, char * const units);
double globalRecvTest(const int id, const int rank, const int param, char * const units);
double globalSendRecvTest(const int id, const int rank, const int param, char * const units);
double globalBarrierTest(const int id, const int rank, const int param, char * const units);
double globalBroadcastSendTest(const int id, const int rank, const int param, char * const units);
double globalBroadcastRecvTest(const int id, const int rank, const int param, char * const units);
double globalBroadcastRecv2Test(const int id, const int rank, const int param, char * const units);
double globalBroadcastRecv3Test(const int id, const int rank, const int param, char * const units);
double globalBroadcastRecv4Test(const int id, const int rank, const int param, char * const units);

int main(int argc, char ** argv)
{
  int gpus[] = { -1 };
  dcgn::initAll(&argc, &argv, 2, gpus, 0, 0, 0);

  dcgn::launchCPUKernel(0, timer, 0);
  dcgn::launchCPUKernel(1, timer, 0);

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

void timer(void * )
{
  SpeedTest tests[] =
  {
    localSendRecvTest,

    globalSendTest,
    globalRecvTest,
    globalSendRecvTest,
    globalBarrierTest,
    globalBroadcastSendTest,
    globalBroadcastRecvTest,
    globalBroadcastRecv2Test,
    globalBroadcastRecv3Test,
    globalBroadcastRecv4Test,
  };
  const char * const testNames[] =
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
  const int NUM_TESTS = sizeof(tests) / sizeof(tests[0]);
  const int NUM_UNITS = 17;
  const int NUM_ITERS = 30;
  double runTimes[NUM_ITERS];

  int id, size;
  id    = dcgn::getRank();
  size  = dcgn::getSize();

  dcgn::barrier();
  for (int i = 0; i < NUM_TESTS; ++i)
  {
    if (id == 0)
    {
      printf("%-25s - %3d CPUs.\n", testNames[i], size);
      printf("           | Min. Time   | Max. Time   | Mean Time   | Median Time |\n");
      printf("+----------+-------------+-------------+-------------+-------------+\n");
    }
    char units[40];
    for (int j = 0; j < NUM_UNITS; ++j)
    {
      for (int k = 0; k < NUM_ITERS; ++k)
      {
        runTimes[k] = tests[i](id, size, j, units);
      }
      qsort(runTimes, NUM_ITERS, sizeof(double), dcmp);
      double avg = average(runTimes, NUM_ITERS);
      double sd = stddev(runTimes, NUM_ITERS);
      int lowIndex = 0, hiIndex = NUM_ITERS - 1;
      while ((runTimes[lowIndex] < avg - 2 * sd) && lowIndex < NUM_ITERS) ++lowIndex;
      while ((runTimes[hiIndex ] > avg + 2 * sd) && hiIndex  > lowIndex)  --hiIndex;
      avg = average(runTimes + lowIndex, hiIndex - lowIndex + 1);
      if (id == 0)
      {
        printf("| %-8s | %11f | %11f | %11f | %11f | %d\n", units, runTimes[lowIndex], runTimes[hiIndex], avg, runTimes[lowIndex + (hiIndex - lowIndex) / 2], hiIndex - lowIndex + 1);
        fflush(stdout);
      }
    }
    if (id == 0)
    {
      printf("+----------+-------------+-------------+-------------+-------------+\n");
      fflush(stdout);
    }
  }
}

double localSendRecvTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 0)
  {
    dcgn::CommStatus stat;
    char * buf = new char[amt];
    double t = dcgn::wallTime();
    dcgn::sendRecvReplace(0, 0, buf, amt, &stat);
    t = dcgn::wallTime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double globalSendTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 0)
  {
    char * buf = new char[amt];
    double t = dcgn::wallTime();
    dcgn::send(2, buf, amt);
    t = dcgn::wallTime() - t;
    delete [] buf;
    return t;
  }
  else if (id == 2)
  {
    dcgn::CommStatus stat;
    char * buf = new char[amt];
    double t = dcgn::wallTime();
    dcgn::recv(0, buf, amt, &stat);
    t = dcgn::wallTime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double globalRecvTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 2)
  {
    char * buf = new char[amt];
    double t = dcgn::wallTime();
    dcgn::send(0, buf, amt);
    t = dcgn::wallTime() - t;
    delete [] buf;
    return t;
  }
  else if (id == 0)
  {
    dcgn::CommStatus stat;
    char * buf = new char[amt];
    double t = dcgn::wallTime();
    dcgn::recv(2, buf, amt, &stat);
    t = dcgn::wallTime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double globalSendRecvTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 0 || id == 2)
  {
    dcgn::CommStatus stat;
    char * buf = new char[amt];
    double t = dcgn::wallTime();
    dcgn::sendRecvReplace(2 - id, 2 - id, buf, amt, &stat);
    t = dcgn::wallTime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double globalBarrierTest(const int id, const int rank, const int param, char * const units)
{
  sprintf(units, "%d", 100);
  double t = dcgn::wallTime();
  for (int i = 0; i < 100; ++i)
  {
    dcgn::barrier();
  }
  return dcgn::wallTime() - t;
}

double globalBroadcastSendTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = dcgn::wallTime();
  dcgn::broadcast(0, buf, amt);
  return dcgn::wallTime() - t;
}

double globalBroadcastRecvTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = dcgn::wallTime();
  dcgn::broadcast(1, buf, amt);
  return dcgn::wallTime() - t;
}

double globalBroadcastRecv2Test(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = dcgn::wallTime();
  dcgn::broadcast(2, buf, amt);
  return dcgn::wallTime() - t;
}

double globalBroadcastRecv3Test(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = dcgn::wallTime();
  dcgn::broadcast(3, buf, amt);
  return dcgn::wallTime() - t;
}

double globalBroadcastRecv4Test(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = dcgn::wallTime();
  dcgn::broadcast(6, buf, amt);
  return dcgn::wallTime() - t;
}
