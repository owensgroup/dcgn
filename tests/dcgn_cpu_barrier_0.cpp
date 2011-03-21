#include <dcgn/dcgn.h>
#include <cstdio>

const int MAX_SIZE = 1048576 * 4;
const int ITERS = 30;

void kernel(void * )
{
  dcgn::barrier();

  double t = dcgn::wallTime();
  for (int i = 0; i < 10000; ++i)
  {
    dcgn::barrier();
  }
  t = dcgn::wallTime() - t;
  if (dcgn::getRank() == 0)
  {
    printf("%.6f\n", (t / 10000) * 1000);
    fflush(stdout);
  }
}

int main(int argc, char ** argv)
{
  dcgn::init(&argc, &argv);
  dcgn::initCPU(1);
  dcgn::initComm(-1);
  dcgn::start();

  dcgn::launchCPUKernel(0, kernel, 0);

  dcgn::finalize();

  return 0;
}
