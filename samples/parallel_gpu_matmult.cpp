#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

void setupDevice(const int deviceNum, const int matrixSize);
void mult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C);
void matrixMult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C);
void cleanupDevice();

void sendrecv(int id, void * buf, int count, int dst, int src)
{
  MPI_Status stat;
  MPI_Request req0, req1;

  void * sendBuf = new char[count];
  memcpy(sendBuf, buf, count);
  MPI_Isend(sendBuf,  count, MPI_BYTE, dst, (dst << 16) | id,  MPI_COMM_WORLD, &req0);
  MPI_Irecv(buf,      count, MPI_BYTE, src, (id  << 16) | src, MPI_COMM_WORLD, &req1);
  MPI_Wait(&req0, &stat);
  MPI_Wait(&req1, &stat);
  delete (char * )sendBuf;
}

void doWork(const int matrixSize, const int id, const int subSize,
            const int left, const int right, const int up, const int down,
            float * const A, float * const B, float * const C)
{
  for (int i = 0; i < subSize; ++i)
  {
    // MPI_Status stat;
    mult(matrixSize, matrixSize, matrixSize, A, B, C);

    // MPI_Sendrecv_replace(A, sizeof(float) * matrixSize * matrixSize, MPI_BYTE, left,  (left << 16) | id, right, (id << 16) | right, MPI_COMM_WORLD, &stat);
    // MPI_Sendrecv_replace(B, sizeof(float) * matrixSize * matrixSize, MPI_BYTE, up,    (up   << 16) | id, down,  (id << 16) | down,  MPI_COMM_WORLD, &stat);
    sendrecv(id, A, sizeof(float) * matrixSize * matrixSize, left,  right);
    sendrecv(id, B, sizeof(float) * matrixSize * matrixSize, up,    down);
  }
}

void allocateAndSetup(const int dim, float *& A, float *& B, float *& C, MPI_Request * reqs)
{
  int id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  A = new float[dim * dim];
  B = new float[dim * dim];
  C = new float[dim * dim];
  MPI_Irecv(A, dim * dim * sizeof(float), MPI_BYTE, 0, (id << 16) | 0, MPI_COMM_WORLD, reqs + 0);
  MPI_Irecv(B, dim * dim * sizeof(float), MPI_BYTE, 0, (id << 16) | 0, MPI_COMM_WORLD, reqs + 1);
  memset(C, 0, sizeof(float) * dim * dim);
}

void readMatrices(const char * const file1, const char * const file2, const int subSize, int & dim, float *& A, float *& B, float *& C)
{
  FILE * fp1, * fp2;
  int M, N, N2, P;

  fp1 = fopen(file1, "rb");
  if (!fp1)
  {
    fprintf(stderr, "Error, couldn't open '%s' for reading.\n", file1);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  fp2 = fopen(file2, "rb");
  if (!fp2)
  {
    fclose(fp1);
    fprintf(stderr, "Error, couldn't open '%s' for reading.\n", file2);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  fread(&M,  sizeof(int), 1, fp1);
  fread(&N,  sizeof(int), 1, fp1);
  fread(&N2, sizeof(int), 1, fp2);
  fread(&P,  sizeof(int), 1, fp2);
  if (M != N || M != N2 || M != P)
  {
    fclose(fp1);
    fclose(fp2);
    fprintf(stderr, "Error, matrices must be square.\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  if (M % subSize != 0)
  {
    fclose(fp1);
    fclose(fp2);
    fprintf(stderr, "Error, matrix of size %d x %dis not fittable into processor mapping of size %d x %d\n", dim, dim, subSize, subSize);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  dim = M;
  MPI_Bcast(&dim, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  int size;
  const int subMatDim = dim / subSize;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Status stat;
  MPI_Request reqs[2];
  allocateAndSetup(dim / subSize, A, B, C, reqs);

  float * sendData = new float[subMatDim * subMatDim];

  for (int i = 0; i < size; ++i)
  {
    int r = i / subSize;
    int c = r + i % subSize;
    if (c >= subSize) c -= subSize;
    for (int j = 0; j < subMatDim; ++j)
    {
      fseek(fp1, sizeof(int) * 2 + (r + j) * M * sizeof(float) + sizeof(float) * c, SEEK_SET);
      fread(sendData + subMatDim * j, sizeof(float) * subMatDim, 1, fp1);
    }
    MPI_Send(sendData, subMatDim * subMatDim * sizeof(float), MPI_BYTE, i, (i << 16) | 0, MPI_COMM_WORLD);

    c = i % subSize;
    r = c + i / subSize;
    if (r >= subSize) r -= subSize;
    for (int j = 0; j < subMatDim; ++j)
    {
      fseek(fp2, sizeof(int) * 2 + (r + j) * M * sizeof(float) + sizeof(int) * c, SEEK_SET);
      fread(sendData + subMatDim * j, sizeof(float) * subMatDim, 1, fp2);
    }
    MPI_Send(sendData, subMatDim * subMatDim * sizeof(float), MPI_BYTE, i, (i << 16) | 0, MPI_COMM_WORLD);
  }
  MPI_Wait(&reqs[0], &stat);
  MPI_Wait(&reqs[1], &stat);

  fclose(fp1);
  fclose(fp2);

  delete [] sendData;
}

void recvMatrices(int & dim, const int subSize, float *& A, float *& B, float *& C)
{
  MPI_Bcast(&dim, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Request reqs[2];
  MPI_Status stat;
  allocateAndSetup(dim / subSize, A, B, C, reqs);
  MPI_Wait(&reqs[0], &stat);
  MPI_Wait(&reqs[1], &stat);
}

void calcNeighbors(const int subSize, const int id, int & left, int & right, int & up, int & down)
{
  const int r   = id / subSize, c = id % subSize;
  left  =  (c == 0            ? subSize : c) - 1  + subSize * r;
  right =  (c == subSize - 1  ? -1      : c) + 1  + subSize * r;
  up    = ((r == 0            ? subSize : r) - 1) * subSize + c;
  down  = ((r == subSize - 1  ? -1      : r) + 1) * subSize + c;
}

int main(int argc, char ** argv)
{
  int id, size;
  float * A, * B, * C;
  int dim, subSize = 0;
  int left, right, up, down;
  double start, stop;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  switch (size)
  {
  case   1: subSize =  1; break;
  case   4: subSize =  2; break;
  case   9: subSize =  3; break;
  case  16: subSize =  4; break;
  case  25: subSize =  5; break;
  case  36: subSize =  6; break;
  case  49: subSize =  7; break;
  case  64: subSize =  8; break;
  case  81: subSize =  9; break;
  case 100: subSize = 10; break;
  default:
    if (id == 0)
    {
      fprintf(stderr, "Error, must have a square number of processors.\n");
      fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 0);
    break;
  }

  // if (id == 0) { fprintf(stderr, "setting up world.\n"); fflush(stderr); }
  if (id == 0)  readMatrices(argv[1], argv[2], subSize, dim, A, B, C);
  else          recvMatrices(dim, subSize, A, B, C);
  // if (id == 0) { fprintf(stderr, "calculating neighbors.\n"); fflush(stderr); }
  calcNeighbors(subSize, id, left, right, up, down);
  // if (id == 0) { fprintf(stderr, "setting up device.\n"); fflush(stderr); }
  setupDevice(id / 2, subSize);

  // if (id == 0) { fprintf(stderr, "doing work.\n"); fflush(stderr); }
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  doWork(dim / subSize, id, subSize, left, right, up, down, A, B, C);
  MPI_Barrier(MPI_COMM_WORLD);
  stop = MPI_Wtime();
  if (id == 0) { fprintf(stderr, "done. took %.3f seconds.\n", stop - start); fflush(stderr); }
  delete [] A;
  delete [] B;
  delete [] C;
  cleanupDevice();

  MPI_Finalize();

  return 0;
}
