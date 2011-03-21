#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

const int ITERS = 1000;

void busySleep10ms()
{
  double t = MPI_Wtime();
  while (MPI_Wtime() - t < 0.01) { }
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

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Barrier(MPI_COMM_WORLD);
  double * times = new double[ITERS];
  int iters = ITERS;

  double t = 0;
  for (int i = 0; i < ITERS; ++i)
  {
    if (rank == 0) busySleep10ms();
    double t2 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    times[i] = MPI_Wtime() - t2;
    t += times[i];
    if (rank == 0) busySleep10ms();
  }
  if (rank == 0)
  {
    double sum;
    eliminateOutliers(times, sum, iters);
    printf(" %d iterations took an average of %.8f sec, adjusted (%d samples) %.8f sec\n", ITERS, (t / ITERS), iters, sum / iters);
    fflush(stdout);
  }
  MPI_Finalize();

  return 0;
}
