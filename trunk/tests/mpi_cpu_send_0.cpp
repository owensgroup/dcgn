#include <mpi.h>
#include <cstdlib>
#include <cstdio>

const int NUM_ITERS = 1000;

void cpuKernel(void * ds)
{
  int id;
  int dataSize = (int)(long long)ds;
  char * x = new char[dataSize];
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  MPI_Barrier(MPI_COMM_WORLD);
  double t = MPI_Wtime();
  for (int i = 0; i < NUM_ITERS; ++i)
  {
    MPI_Status stat;
    if (id == 0)  MPI_Send(x, dataSize, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    else          MPI_Recv(x, dataSize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &stat);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t = MPI_Wtime() - t;
  delete [] x;

  if (id == 0)
  {
    printf("%11d - %20.10f ms\n", dataSize, t / NUM_ITERS * 1000.0f);
    fflush(stdout);
  }
}

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  for (int x = 1; x <= 1048576; x *= 2)
  {
    cpuKernel((void * )x);
  }

  MPI_Finalize();

  return 0;
}
