#include <dcgn/dcgn.h>
#include <cstdio>

const int NUM_COMM = 10;

void cpuKernel1(void * )
{
  int x[NUM_COMM];
  dcgn::AsyncRequest reqs[NUM_COMM];
  for (int i = 0; i < NUM_COMM; ++i) x[i] = i + 1;
  for (int i = 0; i < NUM_COMM; ++i)
  {
    dcgn::asyncSend(1, x + i, sizeof(int), reqs + i);
  }
  for (int i = 0; i < NUM_COMM; ++i)
  {
    dcgn::asyncWait(reqs + i, 0);
  }
}

void cpuKernel2(void * )
{
  dcgn::AsyncRequest reqs[NUM_COMM];
  int x[NUM_COMM];

  for (int i = 0; i < NUM_COMM; ++i)
  {
    dcgn::asyncRecv(0, x + i, sizeof(int), reqs + i);
  }
  for (int i = 0; i < NUM_COMM; ++i)
  {
    dcgn::CommStatus stat;
    dcgn::asyncWait(reqs + i, &stat);
    printf("%d\n", *(x + i));
  }
}

int main(int argc, char ** argv)
{
  int gpus[] = { -1 };

  dcgn::initAll(&argc, &argv, 2, gpus, 1, 0, -1);

  dcgn::launchCPUKernel(0, cpuKernel1, 0);
  dcgn::launchCPUKernel(1, cpuKernel2, 0);

  dcgn::finalize();

  return 0;
}
