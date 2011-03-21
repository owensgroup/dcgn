#include <dcgn/dcgn.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

const int ITERS = 1000;

void busySleep10ms()
{
  double t = dcgn::wallTime();
  while (dcgn::wallTime() - t < 0.01) { }
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

void kernel(void * )
{
  dcgn::barrier();

  double * vals = new double[ITERS];
  int iters = ITERS;
  double t = 0;
  for (int i = 0; i < ITERS; ++i)
  {
    if (dcgn::getRank() == 0) busySleep10ms();
    double t2 = dcgn::wallTime();
    dcgn::barrier();
    vals[i] = dcgn::wallTime() - t2;
    t += vals[i];
    if (dcgn::getRank() == 0) busySleep10ms();
  }
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
  dcgn::init(&argc, &argv);
  dcgn::initCPU(2);
  dcgn::initComm(-1);
  dcgn::start();

  dcgn::launchCPUKernel(0, kernel, 0);
  dcgn::launchCPUKernel(1, kernel, 0);

  dcgn::finalize();

  return 0;
}
