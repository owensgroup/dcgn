#include <dcgn/dcgn.h>
#include <cstdio>
#include <cstdlib>

const int MAX_SIZE = 1048576 * 64;
const int ITERS = 30;

void cpuKernel(void * )
{
  char * buf = new char[MAX_SIZE];
  for (int i = 1; i <= MAX_SIZE; i *= 2)
  {
    for (int j = 0; j < 30; ++j) dcgn::barrier();
    double t = dcgn::wallTime();
    for (int j = 0; j < ITERS; ++j)
    {
      dcgn::broadcast(0, buf, i);
    }
    t = dcgn::wallTime() - t;
    if (dcgn::getRank() == 0)
    {
      printf("%3d %2s: %20.6f\n",
             i / (i < 1024 ? 1 : i < 1048576 ? 1024 : 1048576),
             (i < 1024 ? " B" : i < 1048576 ? "kB": "MB"),
             t / ITERS);
    }
  }
  delete [] buf;
}

int main(int argc, char ** argv)
{
  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  dcgn::initCPU(2);
  dcgn::start();

  dcgn::launchCPUKernel(0, cpuKernel, 0);
  dcgn::launchCPUKernel(1, cpuKernel, 0);

  dcgn::finalize();
  return 0;
}
