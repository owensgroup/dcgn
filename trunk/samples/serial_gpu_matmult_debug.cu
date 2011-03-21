// File to perform matrix multiplication on the GPU.

#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#define MAN_DEBUG 0

#define DEBUG_THREAD_BLOCK 0
#define DEBUG_THREAD_ROW 0
#define DEBUG_THREAD_COL 0

#define CHECK_ERR()                                                                                         \
{                                                                                                           \
  cudaError_t err = cudaGetLastError();                                                                     \
  if (err != cudaSuccess)                                                                                   \
  {                                                                                                         \
    fprintf(stderr, "%s.%s.%d: %d, %s.\n", __FILE__, __FUNCTION__, __LINE__, err, cudaGetErrorString(err)); \
    fflush(stderr);                                                                                         \
  }                                                                                                         \
}                                                                                                           \

enum
{
  STRING_SECTION_1 = 0,
  STRING_SECTION_2,
  STRING_SECTION_3,
  STRING_SECTION_4,
  STRING_DELIMITER,
  STRING_M,
  STRING_N,
  STRING_P,
  STRING_A_ROWS,
  STRING_A_COLS,
  STRING_B_COLS,
  STRING_LOCAL_INDEX,
  STRING_LOCAL_ROW,
  STRING_LOCAL_COL,
  STRING_NUM_LOCAL_ROWS,
  STRING_NUM_LOCAL_COLS,
  STRING_C_BLOCK_COLS,
  STRING_C_START_COL,
  STRING_C_START_ROW,
  STRING_I,
  STRING_A_ROW_INDEX,
  STRING_A_COL_INDEX,
  STRING_B_ROW_INDEX,
  STRING_B_COL_INDEX,
  STRING_K,
  STRING_A_BLOCK_ROW,
  STRING_A_BLOCK_COL,
  STRING_B_BLOCK_ROW,
  STRING_B_BLOCK_COL,
  STRING_A_B_BLOCK_MUL,
  STRING_C_PARTIAL,
  STRING_C_ROW,
  STRING_C_COL,
  STRING_C_FINAL,
};

const char * const STRINGS[] =
{
  "Constant Variables",                       // STRING_SECTION_1 = 0,
  "C Block Variables",                        // STRING_SECTION_2,
  "Global Memory Read Indices",               // STRING_SECTION_3,
  "Global Memory Write Indices",              // STRING_SECTION_4,
  "***************************************",  // STRING_DELIMITER,
  "M",                                        // STRING_M,
  "N",                                        // STRING_N,
  "P",                                        // STRING_P,
  "A_ROWS",                                   // STRING_A_ROWS,
  "A_COLS",                                   // STRING_A_COLS,
  "B_COLS",                                   // STRING_B_COLS,
  "LOCAL_INDEX",                              // STRING_LOCAL_INDEX,
  "LOCAL_ROW",                                // STRING_LOCAL_ROW,
  "LOCAL_COL",                                // STRING_LOCAL_COL,
  "NUM_LOCAL_ROW",                            // STRING_NUM_LOCAL_ROWS,
  "NUM_LOCAL_COL",                            // STRING_NUM_LOCAL_COLS,
  "C_BLOCK_COLS",                             // STRING_C_BLOCK_COLS,
  "C_START_COL",                              // STRING_C_START_COL,
  "C_START_ROW",                              // STRING_C_START_ROW,
  "i",                                        // STRING_I,
  "A global read row    index",               // STRING_A_ROW_INDEX,
  "A global read column index",               // STRING_A_COL_INDEX,
  "B global read row    index",               // STRING_B_ROW_INDEX,
  "B global read column index",               // STRING_B_COL_INDEX,
  "k",                                        // STRING_K,
  "A block row    index",                     // STRING_A_BLOCK_ROW,
  "A block column index",                     // STRING_A_BLOCK_COL,
  "B block row    index",                     // STRING_B_BLOCK_ROW,
  "B block column index",                     // STRING_B_BLOCK_COL,
  "A and B block multiply",                   // STRING_A_B_BLOCK_MUL,
  "C so far",                                 // STRING_C_PARTIAL,
  "C global write row index",                 // STRING_C_ROW,
  "C global write column index",              // STRING_C_COL,
  "C final value",                            // STRING_C_FINAL,
};

void dumpMemory(int * const D, int * const E)
{
  for (int i = 0; D[i] != -2; ++i)
  {
    switch (E[i])
    {
    case STRING_SECTION_1:
    case STRING_SECTION_2:
    case STRING_SECTION_3:
    case STRING_SECTION_4:
    case STRING_DELIMITER:
      printf("%s\n", STRINGS[E[i]]);
      break;
    case STRING_A_B_BLOCK_MUL:
    case STRING_C_PARTIAL:
    case STRING_C_FINAL:
      printf("%-30s : %10.6f\n", STRINGS[E[i]], *reinterpret_cast<float * >(D + i));
      break;
    default:
      printf("%-30s : %10d\n", STRINGS[E[i]], D[i]);
      break;
    }
  }
}

// A is an MxN matrix, B is an NxP matrix, and C is an MxP matrix.
#if MAN_DEBUG
  __global__ void matrixMult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C, int * const D, int * const E);
#else
  __global__ void matrixMult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C);
#endif

int main(int argc, char ** argv)
{
  FILE * in1, * in2, * out;
  float * A, * B, * C;
#if MAN_DEBUG
  int * D, * E;
#endif
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
#if MAN_DEBUG
  D = reinterpret_cast<int   * >(malloc(sizeof(int)   * 1048576));
  E = reinterpret_cast<int   * >(malloc(sizeof(int)   * 1048576));
#endif

  // read the input matrices into memory.
  fread(A, sizeof(float) * M * N, 1, in1);
  fread(B, sizeof(float) * N * P, 1, in2);
  fclose(in1);
  fclose(in2);

  float * gpuA, * gpuB, * gpuC;
#if MAN_DEBUG
  int * gpuD, * gpuE;
#endif

  // allocate the gpu matrices.
  cudaMalloc(reinterpret_cast<void ** >(&gpuA), sizeof(float) * M * N);   CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuB), sizeof(float) * N * P);   CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuC), sizeof(float) * M * P);   CHECK_ERR();
#if MAN_DEBUG
  cudaMalloc(reinterpret_cast<void ** >(&gpuD), sizeof(int)   * 1048576); CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuE), sizeof(int)   * 1048576); CHECK_ERR();
#endif

  // copy the memory matrices down to the gpu.
  cudaMemcpy(reinterpret_cast<void * >(gpuA), reinterpret_cast<void * >(A),    sizeof(float) * M * N, cudaMemcpyHostToDevice); CHECK_ERR();
  cudaMemcpy(reinterpret_cast<void * >(gpuB), reinterpret_cast<void * >(B),    sizeof(float) * N * P, cudaMemcpyHostToDevice); CHECK_ERR();

  uint3 gs = { 12, 1, 1 }, bs = { 16, 16, 1 };

  // launch the kernel. and wait for it to finish.
#if MAN_DEBUG
  matrixMult<<<gs, bs>>>(M, N, P, gpuA, gpuB, gpuC, gpuD, gpuE);  CHECK_ERR();
#else
  matrixMult<<<gs, bs>>>(M, N, P, gpuA, gpuB, gpuC);              CHECK_ERR();
#endif
  cudaThreadSynchronize();                                        CHECK_ERR();

  // copy the resulting matrix back up from the gpu.
  cudaMemcpy(reinterpret_cast<void * >(C),    reinterpret_cast<void * >(gpuC), sizeof(float) * M * P, cudaMemcpyDeviceToHost); CHECK_ERR();
#if MAN_DEBUG
  cudaMemcpy(reinterpret_cast<void * >(D),    reinterpret_cast<void * >(gpuD), sizeof(int)   * 1048576,  cudaMemcpyDeviceToHost); CHECK_ERR();
  cudaMemcpy(reinterpret_cast<void * >(E),    reinterpret_cast<void * >(gpuE), sizeof(int)   * 1048576,  cudaMemcpyDeviceToHost); CHECK_ERR();
#endif

  // clean up the gpu memory.
  cudaFree(gpuA); CHECK_ERR();
  cudaFree(gpuB); CHECK_ERR();
  cudaFree(gpuC); CHECK_ERR();
#if MAN_DEBUG
  cudaFree(gpuD); CHECK_ERR();
  cudaFree(gpuE); CHECK_ERR();
#endif

  // write the matrix to file, then shut down the file.
  fwrite(&M, sizeof(int), 1, out);
  fwrite(&P, sizeof(int), 1, out);
  fwrite(C, sizeof(float) * M * P, 1, out);
  fclose(out);

#if MAN_DEBUG
  dumpMemory(D, E);
#endif

  // free up the cpu memory.
  free(A);
  free(B);
  free(C);
#if MAN_DEBUG
  free(D);
  free(E);
#endif

  return 0;
}

__shared__ float mem[2048];

#if MAN_DEBUG
__global__ void matrixMult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C, int * const D, int * const E)
#else
__global__ void matrixMult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C)
#endif
{
  //  basic algorithm
  //  Divide up the matrices A, B, and C into a bunch of sub blocks for computation.
  //  For each block Cij
  //    Loop and grab blocks Aik and Bkj
  //      Load Aki and Bkj into shared memory
  //        Each thread performs a dot product and adds a partial value for Ci
  //    Each thread writes a final value for Ci

  // only handle cases where the matrix blocks will map nicely to thread blocks.
  if (M % blockDim.y != 0 || N % blockDim.x != 0 || N % blockDim.y != 0 || P % blockDim.x != 0)
  {
    return;
  }
  // only handle cases where the blocksize is square. this is to better guarantee fast, coalesced reads.
  if (blockDim.x != blockDim.y || blockDim.z != 1)
  {
    return;
  }

  const int A_ROWS = M, A_COLS = N, B_COLS = P;
  const int LOCAL_ROW = threadIdx.y;
  const int LOCAL_COL = threadIdx.x;
  const int NUM_LOCAL_ROWS = blockDim.y;
  const int NUM_LOCAL_COLS = blockDim.x;
  const int NUM_BLOCKS_C = (A_ROWS / NUM_LOCAL_ROWS) * (B_COLS / NUM_LOCAL_COLS);
  const int LOCAL_INDEX = LOCAL_ROW * NUM_LOCAL_COLS + LOCAL_COL;
  const int C_BLOCK_COLS = B_COLS / NUM_LOCAL_COLS;
  int blockIndexC = blockIdx.x;

  // set some pointers into shared memory for the local copies of A, B, and C.
  float * blockA = mem;
  float * blockB = blockA + NUM_LOCAL_ROWS * NUM_LOCAL_COLS;
  float * blockC = blockB + NUM_LOCAL_ROWS * NUM_LOCAL_COLS;

#if MAN_DEBUG
  int dindex = 0, eindex = 0;

  if (LOCAL_ROW == DEBUG_THREAD_ROW && LOCAL_COL == DEBUG_THREAD_COL && blockIdx.x == DEBUG_THREAD_BLOCK)
  {
    D[dindex++] = 10000;          E[eindex++] = STRING_SECTION_1;
    D[dindex++] = N;              E[eindex++] = STRING_N;
    D[dindex++] = A_ROWS;         E[eindex++] = STRING_A_ROWS;
    D[dindex++] = A_COLS;         E[eindex++] = STRING_A_COLS;
    D[dindex++] = B_COLS;         E[eindex++] = STRING_B_COLS;
    D[dindex++] = LOCAL_ROW;      E[eindex++] = STRING_LOCAL_ROW;
    D[dindex++] = LOCAL_COL;      E[eindex++] = STRING_LOCAL_COL;
    D[dindex++] = NUM_LOCAL_ROWS; E[eindex++] = STRING_NUM_LOCAL_ROWS;
    D[dindex++] = NUM_LOCAL_COLS; E[eindex++] = STRING_NUM_LOCAL_COLS;
    D[dindex++] = LOCAL_INDEX;    E[eindex++] = STRING_LOCAL_INDEX;
    D[dindex++] = C_BLOCK_COLS;   E[eindex++] = STRING_C_BLOCK_COLS;
    D[dindex++] = -1;             E[eindex++] = STRING_DELIMITER;
    D[dindex++] = -1;             E[eindex++] = STRING_DELIMITER;
    D[dindex++] = -1;             E[eindex++] = STRING_DELIMITER;
  }
#endif

  // loop for every sub-matrix Ci that we can.
  while (blockIndexC < NUM_BLOCKS_C)
  {
    // the upper left corner in C of Ci.
    const int C_START_COL = blockIndexC % C_BLOCK_COLS; // the starting column of our sub-block of C.
    const int C_START_ROW = blockIndexC / C_BLOCK_COLS; // the starting row of our sub-block of C.

#if MAN_DEBUG
    if (LOCAL_ROW == DEBUG_THREAD_ROW && LOCAL_COL == DEBUG_THREAD_COL && blockIdx.x == DEBUG_THREAD_BLOCK)
    {
      D[dindex++] = 20000;        E[eindex++] = STRING_SECTION_2;
      D[dindex++] = C_START_COL;  E[eindex++] = STRING_C_START_COL;
      D[dindex++] = C_START_ROW;  E[eindex++] = STRING_C_START_ROW;
    }
#endif

    // set the initial result of the sub matrix to 0.
    blockC[LOCAL_INDEX] = 0.0f;

    // go through the width of A and height of B to grab the necessary sub blocks.
    for (int i = 0; i < N; i += NUM_LOCAL_COLS)
    {
#if MAN_DEBUG
      if (LOCAL_ROW == DEBUG_THREAD_ROW && LOCAL_COL == DEBUG_THREAD_COL && blockIdx.x == DEBUG_THREAD_BLOCK)
      {
        D[dindex++] = -1;                                       E[eindex++] = STRING_DELIMITER;
        D[dindex++] = -1;                                       E[eindex++] = STRING_DELIMITER;
        D[dindex++] = -1;                                       E[eindex++] = STRING_DELIMITER;
        D[dindex++] = 30000;                                    E[eindex++] = STRING_SECTION_3;
        D[dindex++] = i;                                        E[eindex++] = STRING_I,
        D[dindex++] = C_START_ROW * NUM_LOCAL_ROWS + LOCAL_ROW; E[eindex++] = STRING_A_ROW_INDEX;
        D[dindex++] = i + LOCAL_COL;                            E[eindex++] = STRING_A_COL_INDEX;
        D[dindex++] = i           + LOCAL_ROW;                  E[eindex++] = STRING_B_ROW_INDEX;
        D[dindex++] = C_START_COL * NUM_LOCAL_COLS + LOCAL_COL; E[eindex++] = STRING_B_COL_INDEX;
        D[dindex++] = -1;                                       E[eindex++] = STRING_DELIMITER;
      }
#endif

      // make sure each block is grabbed efficiently
      blockA[LOCAL_INDEX] = A[(C_START_ROW * NUM_LOCAL_ROWS + LOCAL_ROW) * A_COLS + i                            + LOCAL_COL];
      blockB[LOCAL_INDEX] = B[(i                            + LOCAL_ROW) * B_COLS + C_START_COL * NUM_LOCAL_COLS + LOCAL_COL];
      __syncthreads();
      for (int k = 0; k < NUM_LOCAL_COLS; ++k)
      {
        blockC[LOCAL_INDEX] += blockA[LOCAL_ROW * NUM_LOCAL_COLS + k] * blockB[k * NUM_LOCAL_COLS + LOCAL_COL];
        __syncthreads();
#if MAN_DEBUG
#if 0
        if (LOCAL_ROW == DEBUG_THREAD_ROW && LOCAL_COL == DEBUG_THREAD_COL && blockIdx.x == DEBUG_THREAD_BLOCK)
        {
          float t = blockA[LOCAL_ROW * NUM_LOCAL_COLS + k] * blockB[k * NUM_LOCAL_COLS + LOCAL_COL];
          D[dindex++] = k;                                                  E[eindex++] = STRING_K;
          D[dindex++] = (LOCAL_ROW * NUM_LOCAL_COLS + k) / NUM_LOCAL_COLS;  E[eindex++] = STRING_A_BLOCK_ROW;
          D[dindex++] = (LOCAL_ROW * NUM_LOCAL_COLS + k) % NUM_LOCAL_COLS;  E[eindex++] = STRING_A_BLOCK_COL;
          D[dindex++] = (k * NUM_LOCAL_COLS + LOCAL_COL) / NUM_LOCAL_COLS;  E[eindex++] = STRING_B_BLOCK_ROW;
          D[dindex++] = (k * NUM_LOCAL_COLS + LOCAL_COL) % NUM_LOCAL_COLS;  E[eindex++] = STRING_B_BLOCK_COL;
          D[dindex++] = __float_as_int(t);                                  E[eindex++] = STRING_A_B_BLOCK_MUL;
          D[dindex++] = __float_as_int(blockC[LOCAL_INDEX]);                E[eindex++] = STRING_C_PARTIAL;
          D[dindex++] = -1;                                                 E[eindex++] = STRING_DELIMITER;
        }
#endif
#endif
      }
    }

    C[(C_START_ROW * NUM_LOCAL_ROWS + LOCAL_ROW) * B_COLS + C_START_COL * NUM_LOCAL_COLS + LOCAL_COL] = blockC[LOCAL_INDEX];
#if MAN_DEBUG
    if (LOCAL_ROW == DEBUG_THREAD_ROW && LOCAL_COL == DEBUG_THREAD_COL && blockIdx.x == DEBUG_THREAD_BLOCK)
    {
      D[dindex++] = -1;                                                                               E[eindex++] = STRING_DELIMITER;
      D[dindex++] = -1;                                                                               E[eindex++] = STRING_DELIMITER;
      D[dindex++] = 40000;                                                                            E[eindex++] = STRING_SECTION_4;
      D[dindex++] = C_START_ROW * NUM_LOCAL_ROWS + LOCAL_ROW;                                         E[eindex++] = STRING_C_ROW;
      D[dindex++] = C_START_COL * NUM_LOCAL_COLS + LOCAL_COL;                                         E[eindex++] = STRING_C_COL;
      D[dindex++] = __float_as_int(C[(C_START_ROW + LOCAL_ROW) * B_COLS + C_START_COL + LOCAL_COL]);  E[eindex++] = STRING_C_FINAL;
      D[dindex++] = -1;                                                                               E[eindex++] = STRING_DELIMITER;
      D[dindex++] = -1;                                                                               E[eindex++] = STRING_DELIMITER;
      D[dindex++] = -1;                                                                               E[eindex++] = STRING_DELIMITER;
    }
#endif

    blockIndexC += gridDim.x;
  }
#if MAN_DEBUG
  if (LOCAL_ROW == DEBUG_THREAD_ROW && LOCAL_COL == DEBUG_THREAD_COL && blockIdx.x == DEBUG_THREAD_BLOCK)
  {
    D[dindex++] = E[eindex++] = -2;
  }
#endif
  __syncthreads();
}
