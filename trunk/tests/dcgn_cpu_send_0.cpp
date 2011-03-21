#include <dcgn/dcgn.h>
#include <cstdlib>
#include <cstdio>

const int NUM_ITERS = 1000;

void cpuKernel(void * ds)
{
  int id = dcgn::getRank();
  int dataSize = (int )(long long)ds;
  char * x = new char[dataSize];

  dcgn::barrier();

  double t = dcgn::wallTime();
  for (int i = 0; i < NUM_ITERS; ++i)
  {
    dcgn::CommStatus stat;
    if (id == 0)  dcgn::send(1, x, dataSize);
    else          dcgn::recv(0, x, dataSize, &stat);
  }
  dcgn::barrier();
  t = dcgn::wallTime() - t;

  if (id == 0)
  {
    printf("%11d - %20.10f\n", dataSize, t / NUM_ITERS);
    fflush(stdout);
  }
  delete [] x;
}

int main(int argc, char ** argv)
{
  int cpus = 2;
  int gpus[] = { -1 };
  dcgn::initAll(&argc, &argv, cpus, gpus, 1, 0, -1);

  for (int x = 1; x <= 1048576; x *= 2)
  {
    dcgn::launchCPUKernel(0, cpuKernel, (void * )x);
    dcgn::launchCPUKernel(1, cpuKernel, (void * )x);
  }

  dcgn::finalize();

  return 0;
}
