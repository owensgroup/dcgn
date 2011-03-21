#ifndef __host__
  #define __host__
#endif
#ifndef __device__
  #define __device__
#endif
#ifndef __inline__
  #define __inline__
#endif

#include <mpi.h>
#include "samples/Body.cxx"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sys/time.h>

double getTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1000000.0;
}

int getIndexForRank(const int rank, const int size, const int n)
{
  if (rank == 0) return 0;
  if (rank == size) return n;
  float f = static_cast<float>(rank) / static_cast<float>(size) * n;
  return static_cast<int>(f);
}

int main(int argc, char ** argv)
{
  Body * bodies, * updatedBodies;
  int size, timeSteps;
  double timeDelta;
  FILE * fp = 0, * outfp = 0;

  MPI_Init(&argc, &argv);

  int commRank, commSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);

  if (commRank == 0)
  {
    if (argc != 5)
    {
      fprintf(stderr, "Usage: %s <input_file> <output_file> <time_steps> <time_delta>\n", argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 0);
      return 1;
    }
    fp = fopen(argv[1], "rb");
    if (!fp)
    {
      fprintf(stderr, "Couldn't open %s for reading\n", argv[1]);
      MPI_Abort(MPI_COMM_WORLD, 0);
      return 1;
    }
    outfp = fopen(argv[2], "wb");
    if (!outfp)
    {
      fclose(fp);
      fprintf(stderr, "Couldn't open %s for writing\n", argv[2]);
      MPI_Abort(MPI_COMM_WORLD, 0);
      return 1;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sscanf(argv[3], "%d", &timeSteps);
  sscanf(argv[4], "%lf", &timeDelta);

  if (commRank == 0)
  {
    int dummy;
    fread(&size, sizeof(int), 1, fp);
    fread(&dummy, sizeof(int), 1, fp);
  }
  MPI_Bcast(&size, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  bodies = new Body[size];
  updatedBodies = new Body[size];
  if (commRank == 0)
  {
    fread(bodies, sizeof(Body) * size, 1, fp);
    fclose(fp);

    fwrite(&size,       sizeof(int), 1, outfp);
    fwrite(&timeSteps,  sizeof(int), 1, outfp);
  }

  MPI_Bcast(bodies, sizeof(Body) * size, MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  double start = getTime();

  const int START_INDEX = getIndexForRank(commRank,     commSize, size);
  const int END_INDEX   = getIndexForRank(commRank + 1, commSize, size);
  for (int i = 0; i < timeSteps; ++i)
  {
    for (int j = START_INDEX; j < END_INDEX; ++j)
    {
      updatedBodies[j] = bodies[j];
      for (int k = 0; k < size; ++k)
      {
        if (j != k)
        {
          updatedBodies[j].addForceFrom(bodies[k]);
        }
      }
      updatedBodies[j].update(timeDelta);
    }
    for (int j = 0; j < commSize; ++j)
    {
      const int START = getIndexForRank(j,      commSize, size);
      const int END   = getIndexForRank(j + 1,  commSize, size);
      // printf("%2d: broadcast from %d, %d, %d, %d bodies.\n", commRank, j, START, END, END - START);
      MPI_Bcast(updatedBodies + START, sizeof(Body) * (END - START), MPI_BYTE, j, MPI_COMM_WORLD);
    }
    if (commRank == 0)
    {
      // printf("%5d { %g %g %g }\n", i, updatedBodies[0].x, updatedBodies[0].y, updatedBodies[0].z);
      fwrite(updatedBodies, sizeof(Body) * size, 1, outfp);
    }
    if (i + 1 < timeSteps) memcpy(bodies, updatedBodies, sizeof(Body) * size);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (commRank == 0) fclose(outfp);

  double time = getTime() - start;

  if (commRank == 0)
  {
    printf("done, took %.3fs.\n", time);
    fflush(stdout);
  }

  delete [] bodies;
  delete [] updatedBodies;

  MPI_Finalize();
  return 0;
}
