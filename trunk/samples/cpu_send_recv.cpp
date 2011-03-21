#include <dcgn/dcgn.h>
#include <cstdio>
#include <sched.h>

void cpuKernel(void * )
{
  int id = dcgn::getRank(), x = id, y;
  if (id == 0)
  {
    dcgn::send(id + 1, &x, sizeof(int));
  }
  else if (id == dcgn::getSize() - 1)
  {
    dcgn::recv(id - 1, &y, sizeof(int), 0);
    x += y;
  }
  else
  {
    dcgn::recv(id - 1, &y, sizeof(int), 0);
    x += y;
    dcgn::send(id + 1, &x, sizeof(int));
  }
  dcgn::barrier();
  printf("%-5.3d - %5.2d\n", id, x);
  fflush(stdout);
}

int main(int argc, char ** argv)
{
  // int cpus[] = { 0, 1 };
  int gpus[] = { -1 };
  // uint3 gridSize  = { 32, 1, 1 };
  // uint3 blockSize = { 32, 32, 1 };

  dcgn::initAll(&argc, &argv, 2, gpus, 1, 0, -1);

  dcgn::launchCPUKernel(0, cpuKernel, 0);
  dcgn::launchCPUKernel(1, cpuKernel, 0);

  while (!dcgn::areAllLocalResourcesIdle())
  {
    sched_yield();
  }

  dcgn::finalize();

  return 0;
}
