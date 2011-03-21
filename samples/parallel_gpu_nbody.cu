#include <mpi.h>
#include <cuda.h>
#include "samples/Body.cxx"
#include <cstdio>
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

#define CHECK_ERR()                                                                               \
{                                                                                                 \
  cudaError_t err = cudaGetLastError();                                                           \
  if (err != cudaSuccess)                                                                         \
  {                                                                                               \
    fprintf(stderr, "%s.%s.%d: %s\n", __FILE__, __FUNCTION__, __LINE__, cudaGetErrorString(err)); \
    fflush(stderr);                                                                               \
    exit(1);                                                                                      \
  }                                                                                               \
}                                                                                                 \

__shared__ char mem[16 * 1024 - 256]; // 16k - 256bytes for some system stuff.

__device__ void block_memcpy(void * dstMem, const void * srcMem, const int size)
{
        char * dst = reinterpret_cast<       char * >(dstMem);
  const char * src = reinterpret_cast<const  char * >(srcMem);
  int rem = size;

  dst += threadIdx.x * sizeof(float);
  src += threadIdx.x * sizeof(float);

  __syncthreads();
  while (rem >= blockDim.x * sizeof(float))
  {
    *reinterpret_cast<float * >(dst) = *reinterpret_cast<const float * >(src);
    dst += blockDim.x * sizeof(float);
    src += blockDim.x * sizeof(float);
    rem    -= blockDim.x * sizeof(float);
  }
  if (rem > 0 && threadIdx.x == 0)
  {
    while (rem > sizeof(float))
    {
      reinterpret_cast<float * >(dst)[threadIdx.x] = *reinterpret_cast<const float * >(src);
      dst += sizeof(float);
      src += sizeof(float);
      rem    -= sizeof(float);
    }
    while (rem > sizeof(char))
    {
      *(dst++) = *(src++);
      --rem;
    }
  }
  __syncthreads();
}

__device__ void block_memcpy2(void * dstMem, const void * srcMem, const int size)
{
        char * dst = reinterpret_cast<       char * >(dstMem);
  const char * src = reinterpret_cast<const  char * >(srcMem);
  int rem = size;

  dst += threadIdx.x * sizeof(float);
  src += threadIdx.x * sizeof(float);

  __syncthreads();
  while (rem >= blockDim.x * sizeof(float))
  {
    *reinterpret_cast<float * >(dst) = *reinterpret_cast<const float * >(src);
    dst += blockDim.x * sizeof(float);
    src += blockDim.x * sizeof(float);
    rem    -= blockDim.x * sizeof(float);
  }
  if (rem > 0 && threadIdx.x == 0)
  {
    while (rem > sizeof(float))
    {
      reinterpret_cast<float * >(dst)[threadIdx.x] = *reinterpret_cast<const float * >(src);
      dst += sizeof(float);
      src += sizeof(float);
      rem    -= sizeof(float);
    }
    while (rem > sizeof(char))
    {
      *(dst++) = *(src++);
      --rem;
    }
  }
  __syncthreads();
}

__global__ void kernel(Body * bodies, Body * ubodies, const int startBody, const int endBody, const int n, const float timeDelta)
{
  // load up 2 bodies per thread. then load up one body per thread for work.

  const int BODY_START  = startBody;
  const int BODY_STRIDE = blockDim.x * gridDim.x;

  for (int i = BODY_START; i < endBody; i += BODY_STRIDE)
  {
    const int NUM_BODIES = (i + blockDim.x <= endBody ? blockDim.x : endBody - i);
    Body * sharedBodies = reinterpret_cast<Body * >(mem);
    Body * localBodies  = reinterpret_cast<Body * >(mem + NUM_BODIES * sizeof(Body));
    block_memcpy(sharedBodies, bodies + i, NUM_BODIES * sizeof(Body));
    if (threadIdx.x < NUM_BODIES)
    {
      for (int j = 0; j < n; j += blockDim.x * 2)
      {
        const int NUM_BODIES2 = ((j + blockDim.x * 2 <= n) ? (2 * blockDim.x) : (n - j));
        block_memcpy(localBodies, bodies + j, NUM_BODIES2 * sizeof(Body));
        for (int k = 0; k < NUM_BODIES2; ++k)
        {
          sharedBodies[threadIdx.x].addForceFrom(localBodies[k]);
        }
      }
      sharedBodies[threadIdx.x].update(timeDelta);
    }
    block_memcpy2(ubodies + i, sharedBodies, NUM_BODIES * sizeof(Body));
  }
}

__host__ void runKernel(Body * bodies, Body * gpuBodies, Body * gpuUpdatedBodies, const int startBody, const int endBody, const int n, const float timeDelta)
{
  // int id;
  // double t[4];
  uint3 gs = {  12, 1, 1 };
  uint3 bs = { 128, 1, 1 };
  // MPI_Comm_rank(MPI_COMM_WORLD, &id);
  // t[0] = MPI_Wtime();
  cudaMemcpy(gpuBodies, bodies, sizeof(Body) * n, cudaMemcpyHostToDevice);            CHECK_ERR();
  // t[1] = MPI_Wtime();
  kernel<<<gs, bs>>>(gpuBodies, gpuUpdatedBodies, startBody, endBody, n, timeDelta);  CHECK_ERR();
  cudaThreadSynchronize();                                                            CHECK_ERR();
  // t[2] = MPI_Wtime();
  cudaMemcpy(bodies, gpuUpdatedBodies, sizeof(Body) * n, cudaMemcpyDeviceToHost);     CHECK_ERR();
  // t[3] = MPI_Wtime();
  // printf("%2d: cpu->gpu kernel gpu->cpu { %f %f %f } sb eb n { %5d %5d %5d }\n", id, t[1] - t[0], t[2] - t[1], t[3] - t[2], startBody, endBody, endBody - startBody); fflush(stdout);
}

int main(int argc, char ** argv)
{
  Body * bodies, * gpuBodies, * gpuUpdatedBodies;
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
/*
  char buf[40];
  sprintf(buf, "echo %2d `uname -n`", commRank);
  system(buf);
*/
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
  cudaSetDevice(commRank / 4);
  cudaMalloc((void ** )&gpuBodies,        sizeof(Body) * size); CHECK_ERR();
  cudaMalloc((void ** )&gpuUpdatedBodies, sizeof(Body) * size); CHECK_ERR();
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
    double kernelTime = MPI_Wtime();
    runKernel(bodies, gpuBodies, gpuUpdatedBodies, START_INDEX, END_INDEX, size, timeDelta);
    kernelTime = MPI_Wtime() - kernelTime;
    for (int j = 0; j < commSize; ++j)
    {
      const int START = getIndexForRank(j,      commSize, size);
      const int END   = getIndexForRank(j + 1,  commSize, size);
      // printf("%2d: broadcast from %d, %d, %d, %d bodies.\n", commRank, j, START, END, END - START);
      MPI_Bcast(bodies + START, sizeof(Body) * (END - START), MPI_BYTE, j, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (commRank == 0)
    {
      // printf("%5d { %g %g %g }\n", i, updatedBodies[0].x, updatedBodies[0].y, updatedBodies[0].z);
      // fwrite(updatedBodies, sizeof(Body) * size, 1, outfp);
    }
    // if (i + 1 < timeSteps) memcpy(bodies, updatedBodies, sizeof(Body) * size);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (commRank == 0) fclose(outfp);

  double time = getTime() - start;

  if (commRank == 0)
  {
    printf("done, took %f seconds.\n", time);
    fflush(stdout);
  }

  delete [] bodies;
  cudaFree(gpuBodies);
  cudaFree(gpuUpdatedBodies);

  MPI_Finalize();
  return 0;
}
