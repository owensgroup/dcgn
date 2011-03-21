#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <sys/time.h>

#define MAN_DEBUG 0
const int BLOCK_SIZE = 128;

double wallTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (tv.tv_sec * 1000000.0 + tv.tv_usec) / 1000000.0;
}

void matMult(const float * const A, const float * const B, float * const C, const int M)
{
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      float * c = C + i * M + j;
      for (int k = 0; k < M; ++k)
      {
        *c += *(A + i * M + k) * *(B + k * M + j);
      }
    }
  }
}

void loadSubBlock(const float * const A, const int M, float * const subA, const int subRow, const int subCol)
{
  for (int i = 0; i < BLOCK_SIZE; ++i)
  {
    memcpy(subA + BLOCK_SIZE * i, A + M * (subRow * BLOCK_SIZE + i) + subCol * BLOCK_SIZE, sizeof(float) * BLOCK_SIZE);
  }
}

void storeSubBlock(float * const A, const int M, const float * const subA, const int subRow, const int subCol)
{
  for (int i = 0; i < BLOCK_SIZE; ++i)
  {
    memcpy(A + M * (subRow * BLOCK_SIZE + i) + subCol * BLOCK_SIZE, subA + BLOCK_SIZE * i, sizeof(float) * BLOCK_SIZE);
  }
}

int main(int argc, char ** argv)
{
  FILE * in1, * in2, * out;
  float * A, * B, * C;
  int M, N, N2, P;
  if (argc == 2)
  {
    char output[1024];
    if (sscanf(argv[1], "%d,%d,%d,%s", &M, &N, &P, output) == 4)
    {
      fprintf(stderr, "Writing %dx%d matrix to %s.\n", M, N, output);
      out = fopen(output, "wb");
      srand(P);
      C = new float[M * N];
      float * c = C;
      for (int i = M * N - 1; i != -1; --i)
      {
        *(c++) = static_cast<float>(static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
      }
      fwrite(&M, sizeof(int), 1, out);
      fwrite(&N, sizeof(int), 1, out);
      fwrite(C, sizeof(float) * M * N, 1, out);
      fclose(out);
      return 0;
    }
  }
  if (argc == 3)
  {
    in1 = fopen(argv[1], "rb");
    in2 = fopen(argv[2], "rb");
    fread(&M,   sizeof(int), 1, in1);
    fread(&N,   sizeof(int), 1, in1);
    fread(&N2,  sizeof(int), 1, in2);
    fread(&P,   sizeof(int), 1, in2);
    if (M != N2 || N != P)
    {
      fprintf(stderr, "Error, dimensions don't match. (%d, %d) vs (%d, %d).\n", M, N, N2, P);
      fclose(in1);
      fclose(in2);
      return 0;
    }
    A = new float[N];
    B = new float[N];

    memset(A, 0, sizeof(float) * N);
    memset(B, 0, sizeof(float) * N);

    const float BUCKET_RANGES[] =
    {
      0.0f,
      0.000030517578125f,
      0.00006103515625f,
      0.0001220703125f,
      0.000244140625f,
      0.00048828125f,
      0.0009765625f,
      0.001953125f,
      0.00390625f,
      0.0078125f,
      0.015625f,
      0.03125f,
      0.0625f,
      0.125f,
      0.25f,
      0.5f,
      1.0f,
      2.0f,
      4.0f,
      8.0f,
      16.0f,
      32.0f,
      64.0f,
    };
    const int NUM_RANGES = sizeof(BUCKET_RANGES) / sizeof(BUCKET_RANGES[0]);
    int * buckets = new int[NUM_RANGES + 1];
    for (int i = 0; i < M; ++i)
    {
      fread(A, sizeof(float) * N, 1, in1);
      fread(B, sizeof(float) * N, 1, in2);
      for (int j = 0; j < N; ++j)
      {
        float diff = A[j] - B[j];
        // printf("A B diff : { %f %f %f }\n", A[j], B[j], diff);
        if (diff < 0) diff = -diff;
        bool found = false;
        for (int k = 0; !found && k < NUM_RANGES; ++k)
        {
          if (diff <= BUCKET_RANGES[k])
          {
            ++buckets[k];
            found = true;
          }
        }
        if (!found) ++buckets[NUM_RANGES];
      }
    }

    fclose(in1);
    fclose(in2);

    printf("%-*s : %d\n", 43, "Match", buckets[0]);
    for (int i = 1; i < NUM_RANGES; ++i)
    {
      printf("%20.16f - %20.16f : %d\n", BUCKET_RANGES[i - 1], BUCKET_RANGES[i], buckets[i]);
    }
    printf("%20.16f - %20s : %d\n", BUCKET_RANGES[NUM_RANGES - 1], "infinity", buckets[NUM_RANGES]);

    delete [] A;
    delete [] B;
    delete [] buckets;

    return 0;
  }
  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s <MxN matrix input file> <NxP matrix input file> <MxP matrix output file>\n", *argv);
    return 0;
  }

  in1 = fopen(argv[1], "rb");
  if (!in1)
  {
    fprintf(stderr, "Couldn't open input file '%s' for reading.\n", argv[1]);
    return 0;
  }
  in2 = fopen(argv[2], "rb");
  if (!in2)
  {
    fclose(in1);
    fprintf(stderr, "Couldn't open input file '%s' for reading.\n", argv[2]);
    return 0;
  }

  fread(&M,  sizeof(int), 1, in1);
  fread(&N,  sizeof(int), 1, in1);
  fread(&N2, sizeof(int), 1, in2);
  fread(&P,  sizeof(int), 1, in2);
  if (N != N2)
  {
    fprintf(stderr, "Error, second dimension of '%s'(%d) is not equal to first dimension of '%s'(%d). Aborting.\n", argv[1], N, argv[2], N2);
    fclose(in1);
    fclose(in2);
    return 0;
  }
  if (M <= 0 || N <= 0 || P <= 0)
  {
    fprintf(stderr, "Error, matrix dimensions must all be greater than 0. Currently specified matrix dimensions are M=%d, N=%d, P=%d.\n",
                     M, N, P);
    fclose(in1);
    fclose(in2);
    return 0;
  }
  out = fopen(argv[3], "wb");
  if (!out)
  {
    fprintf(stderr, "Couldn't open output file '%s' for writing.\n", argv[3]);
    fclose(in1);
    fclose(in2);
    return 0;
  }

  fwrite(&M, sizeof(int), 1, out);
  fwrite(&P, sizeof(int), 1, out);
  A = new float[M * N];
  B = new float[N * P];
  C = new float[M * P];
  memset(C, 0, sizeof(float) * M * P);

  fread(A, sizeof(float) * M * N, 1, in1);
  fread(B, sizeof(float) * N * P, 1, in2);
  fclose(in1);
  fclose(in2);

  double t1 = wallTime();
  // row major
  if (M <= BLOCK_SIZE)
  {
    matMult(A, B, C, M);
  }
  else
  {
    const int NUM_BLOCKS = M / BLOCK_SIZE;
    float * subA = new float[BLOCK_SIZE * BLOCK_SIZE];
    float * subB = new float[BLOCK_SIZE * BLOCK_SIZE];
    float * subC = new float[BLOCK_SIZE * BLOCK_SIZE];

    for (int sr = 0; sr < NUM_BLOCKS; ++sr)
    {
      for (int sc = 0; sc < NUM_BLOCKS; ++sc)
      {
        loadSubBlock(C, M, subC, sr, sc);
        for (int i = 0; i < NUM_BLOCKS; ++i)
        {
          loadSubBlock(A, M, subA, sr,  i);
          loadSubBlock(B, M, subB,  i, sc);
          matMult(subA, subB, subC, BLOCK_SIZE);
        }
        storeSubBlock(C, M, subC, sr, sc);
      }
    }

    delete [] subA;
    delete [] subB;
    delete [] subC;
  }
  t1 = wallTime() - t1;
  fprintf(stderr, "done. took %.3f seconds.\n", t1);

  fwrite(C, sizeof(float) * M * P, 1, out);
  fclose(out);

  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}
