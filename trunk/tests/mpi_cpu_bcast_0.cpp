#include <mpi.h>
#include <cstdlib>
#include <cstdio>

const int NUM = 20;
const int MIN = 1;
const int MAX = 1 << (NUM - 1);
const int ITERS = 30;

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  int index = 0, rank;
  double times[NUM] = { 0.0 };
  unsigned char * t = new unsigned char[MAX];

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (int size = MIN; size <= MAX; size *= 2)
  {
    times[index] = 0.0;
    for (int i = 0; i < ITERS; ++i)
    {
      if (rank == 0)
      {
        double tt = MPI_Wtime();
        while (MPI_Wtime() > tt && MPI_Wtime() - tt < 0.02) { }
      }
      double t0 = MPI_Wtime();
      MPI_Bcast(t, size, MPI_CHAR, 0, MPI_COMM_WORLD);
      double t1 = MPI_Wtime();
      times[index] += (t1 - t0) / (double)ITERS;
      if (rank == 0)
      {
        double tt = MPI_Wtime();
        while (MPI_Wtime() > tt && MPI_Wtime() - tt < 0.02) { }
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

  MPI_Finalize();

  return 0;
}

