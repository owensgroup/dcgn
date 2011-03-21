// File to perform matrix multiplication on the GPU.

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

float * gpuA, * gpuB, * gpuC;

void setupDevice(const int deviceNum, const int matrixSize);
__host__ void mult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C);
__global__ void matrixMult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C);
void cleanupDevice();

#define CHECK_ERR()                                                                                 \
{                                                                                                   \
  cudaError_t err = cudaGetLastError();                                                             \
  if (err != cudaSuccess)                                                                           \
  {                                                                                                 \
    fprintf(stderr, "%s.%s.%d: %s.\n", __FILE__, __FUNCTION__, __LINE__, cudaGetErrorString(err));  \
    fflush(stderr);                                                                                 \
  }                                                                                                 \
}                                                                                                   \

void setupDevice(const int deviceNum, const int matrixSize)
{
  cudaSetDevice(deviceNum); CHECK_ERR();
  // allocate the gpu matrices.
  cudaMalloc(reinterpret_cast<void ** >(&gpuA), sizeof(float) * matrixSize * matrixSize); CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuB), sizeof(float) * matrixSize * matrixSize); CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuC), sizeof(float) * matrixSize * matrixSize); CHECK_ERR();
}
void cleanupDevice()
{
  cudaFree(gpuA); CHECK_ERR();
  cudaFree(gpuB); CHECK_ERR();
  cudaFree(gpuC); CHECK_ERR();
}

__host__ void mult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C)
{
  uint3 gs = { 12, 1, 1 }, bs = { 16, 16, 1 };
  cudaMemcpy(gpuA, A, sizeof(float) * M * M, cudaMemcpyHostToDevice); CHECK_ERR();
  cudaMemcpy(gpuB, B, sizeof(float) * M * M, cudaMemcpyHostToDevice); CHECK_ERR();
  cudaMemcpy(gpuC, C, sizeof(float) * M * M, cudaMemcpyHostToDevice); CHECK_ERR();
  matrixMult<<<gs, bs>>>(M, N, P, gpuA, gpuB, gpuC);                  CHECK_ERR();
  cudaThreadSynchronize();                                            CHECK_ERR();
  cudaMemcpy(C, gpuC, sizeof(float) * M * M, cudaMemcpyDeviceToHost); CHECK_ERR();
}

__shared__ float mem[1024];

__global__ void matrixMult(const int M, const int N, const int P, const float * const A, const float * const B, float * const C)
{
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

  // loop for every sub-matrix Ci that we can.
  while (blockIndexC < NUM_BLOCKS_C)
  {
    // the upper left corner in C of Ci.
    const int C_START_COL = blockIndexC % C_BLOCK_COLS; // the starting column of our sub-block of C.
    const int C_START_ROW = blockIndexC / C_BLOCK_COLS; // the starting row of our sub-block of C.
    const int C_INDEX = (C_START_ROW * NUM_LOCAL_ROWS + LOCAL_ROW) * B_COLS + C_START_COL * NUM_LOCAL_COLS + LOCAL_COL;

    blockC[LOCAL_INDEX] = C[C_INDEX];
    // blockC[LOCAL_INDEX] = 0.0f;
    // go through the width of A and height of B to grab the necessary sub blocks.
    for (int i = 0; i < N; i += NUM_LOCAL_COLS)
    {
      // make sure each block is grabbed efficiently
      blockA[LOCAL_INDEX] = A[(C_START_ROW * NUM_LOCAL_ROWS + LOCAL_ROW) * A_COLS + i                            + LOCAL_COL];
      blockB[LOCAL_INDEX] = B[(i                            + LOCAL_ROW) * B_COLS + C_START_COL * NUM_LOCAL_COLS + LOCAL_COL];
      __syncthreads();
      float * a = blockA + LOCAL_ROW * NUM_LOCAL_COLS;
      float * b = blockB + LOCAL_COL;
      for (int k = 0; k < NUM_LOCAL_COLS; ++k)
      {
        // blockC[LOCAL_INDEX] += blockA[LOCAL_ROW * NUM_LOCAL_COLS + k] * blockB[k * NUM_LOCAL_COLS + LOCAL_COL];
        blockC[LOCAL_INDEX] = *(a++) * *b;
        b += NUM_LOCAL_COLS;
        __syncthreads();
      }
    }
    C[C_INDEX] = blockC[LOCAL_INDEX];
    blockIndexC += gridDim.x;
  }
  __syncthreads();
}
