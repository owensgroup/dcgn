// File to perform matrix multiplication on the GPU.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

void setupDevice(const int deviceNum, const int matrixSize);
void mult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C);
void matrixMult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C);
void cleanupDevice();

double wallTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (tv.tv_sec * 1000000.0 + tv.tv_usec) / 1000000.0;
}

int main(int argc, char ** argv)
{
  FILE * in1, * in2, * out;
  float * A, * B, * C;
  int M, N, N2, P;
  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s <MxN matrix input file> <NxP matrix input file> <MxP matrix output file>\n", *argv);
    return 0;
  }

  // read the first file.
  in1 = fopen(argv[1], "rb");
  if (!in1)
  {
    fprintf(stderr, "Couldn't open input file '%s' for reading.\n", argv[1]);
    return 0;
  }
  // read the second file.
  in2 = fopen(argv[2], "rb");
  if (!in2)
  {
    fclose(in1);
    fprintf(stderr, "Couldn't open input file '%s' for reading.\n", argv[2]);
    return 0;
  }

  // read the dimensions of the two input matrices.
  fread(&M,  sizeof(int), 1, in1);
  fread(&N,  sizeof(int), 1, in1);
  fread(&N2, sizeof(int), 1, in2);
  fread(&P,  sizeof(int), 1, in2);

  if (M != N || M != N2 || M != P)
  {
    fprintf(stderr, "Error, only square matrices are allowed.\n");
    fflush(stderr);
    fclose(in1);
    fclose(in2);
    return 0;
  }
  // make sure the matrices are compatible for multiplication.
  if (N != N2)
  {
    fprintf(stderr, "Error, second dimension of '%s'(%d) is not equal to first dimension of '%s'(%d). Aborting.\n", argv[1], N, argv[2], N2);
    fclose(in1);
    fclose(in2);
    return 0;
  }
  // make sure the matrices are valid.
  if (M <= 0 || N <= 0 || P <= 0)
  {
    fprintf(stderr, "Error, matrix dimensions must all be greater than 0. Currently specified matrix dimensions are M=%d, N=%d, P=%d.\n",
                     M, N, P);
    fclose(in1);
    fclose(in2);
    return 0;
  }
  // open up the output file.
  out = fopen(argv[3], "wb");
  if (!out)
  {
    fprintf(stderr, "Couldn't open output file '%s' for writing.\n", argv[3]);
    fclose(in1);
    fclose(in2);
    return 0;
  }

  // allocate the cpu memory for the matrices.
  A = reinterpret_cast<float * >(malloc(sizeof(float) * M * N));
  B = reinterpret_cast<float * >(malloc(sizeof(float) * N * P));
  C = reinterpret_cast<float * >(malloc(sizeof(float) * M * P));
  memset(C, 0, sizeof(float) * M * P);

  // read the input matrices into memory.
  fread(A, sizeof(float) * M * N, 1, in1);
  fread(B, sizeof(float) * N * P, 1, in2);
  fclose(in1);
  fclose(in2);

  setupDevice(0, M);
  double t1 = wallTime();
  mult(M, N, P, A, B, C);
  t1 = wallTime() - t1;
  cleanupDevice();

  fprintf(stderr, "done. took %.3f seconds.\n", t1);

  // write the matrix to file, then shut down the file.
  fwrite(&M, sizeof(int), 1, out);
  fwrite(&P, sizeof(int), 1, out);
  fwrite(C, sizeof(float) * M * P, 1, out);
  fclose(out);

  // free up the cpu memory.
  free(A);
  free(B);
  free(C);

  return 0;
}
