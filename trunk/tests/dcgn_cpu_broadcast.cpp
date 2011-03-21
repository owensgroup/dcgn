#include <mpi.h>
#include <cstdio>
#include <cstdlib>

const int NUM = 17;
const int MIN = 1;
const int MAX = 1 << (NUM - 1);
const int ITERS = 30;

void cpuKernel(void * )
{
  int index = 0, rank;
  double times[NUM] = { 0.0 };
  unsigned char * t = new unsigned char[MAX];

  rank = dcgn::getRank();

  for (int size = MIN; size <= MAX; size *= 2)
  {
    times[index] = 0.0;
    for (int i = 0; i < ITERS; ++i)
    {
      if (rank == 0)
      {
        double tt = dcgn::wallTime();
        while (dcgn::wallTime() > tt && dcgn::wallTime() - tt < 0.05) { }
      }
      double t0 = dcgn::wallTime();
      dcgn::bcast(0, t, size);
      double t1 = dcgn::wallTime();
      times[index] += (t1 - t0) / (double)ITERS;
      if (rank == 0)
      {
        double tt = dcgn::wallTime();
        while (dcgn::wallTime() > tt && MPI_Wtime() - tt < 0.05) { }
      }
    }
    if (rank == 0)
    {
      printf("%3d %s: %20f\n",
             size / (size < 1024 ? 1 : size < 1048576 ? 1024 : 1048576),
             size < 1024 ? " B" : size < 1048576 ? "kB" : "MB",
             times[index]);
    }
  }

}

int main(int argc, char ** argv)
{
  char * buf = new char[MAX_SIZE];
  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  dcgn::initCPUs(2);

  dcgn::launchCPUKernel(0, cpuKernel, 0);
  dcgn::launchCPUKernel(1, cpuKernel, 0);

  dcgn::finalize();

  delete [] buf;
  return 0;
}
