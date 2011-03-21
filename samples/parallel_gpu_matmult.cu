#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

int getMatrixDimensions(const char * const file1, const char * const file2);
void matrixMultCPU(void * param);
void matrixMultGPU(void * param, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream);
__global__ void matrixMult(const int subMatDim, const int subSize, float * gpuA, float * gpuB, float * gpuC, const dcgn::GPUInitRequest libParam, volatile int * sbArr);
__device__ void matrixMultSub(const int M, const int N, const int P, const float * const A, const float * const B, float * const C);
__device__ void __syncblocks(volatile int * syncblocksArr);

#define CHECK_ERR()                                                                                 \
{                                                                                                   \
  cudaError_t err = cudaGetLastError();                                                             \
  if (err != cudaSuccess)                                                                           \
  {                                                                                                 \
    fprintf(stderr, "%s.%s.%d: %s.\n", __FILE__, __FUNCTION__, __LINE__, cudaGetErrorString(err));  \
    fflush(stderr);                                                                                 \
  }                                                                                                 \
}                                                                                                   \

double wallTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1000000.0;
}

int main(int argc, char ** argv)
{
  int dim;
  int gpus[] = { 0, 1, -1 };
  uint3 gs = { 12, 1, 1 }, bs = { 16, 16, 1 };

  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  dcgn::initCPU(dcgn::getNodeID() == 0 ? 1 : 0);
  dcgn::initGPU(gpus, 1, 0);
  dcgn::start();

  if (argc != 3)
  {
    fprintf(stderr, "Usage: mpiexec <mpiexec options> %s <matrix1> <matrix2>\n", argv[0]);
    fflush(stderr);
    dcgn::finalize();
    return 0;
  }

  dim = getMatrixDimensions(argv[1], argv[2]);
  if (dim > 0)
  {
    if (dcgn::getNodeID() == 0) dcgn::launchCPUKernel(0, matrixMultCPU, reinterpret_cast<void * >(argv + 1));
    dcgn::launchGPUKernel(0, matrixMultGPU, 0, reinterpret_cast<void * >(&dim), gs, bs, 0);
    dcgn::launchGPUKernel(1, matrixMultGPU, 0, reinterpret_cast<void * >(&dim), gs, bs, 0);
  }

  dcgn::finalize();

  return 0;
}

int getMatrixDimensions(const char * const file1, const char * const file2)
{
  int M, N, N2, P;
  FILE * fp1, * fp2;
  fp1 = fopen(file1, "rb");
  if (!fp1)
  {
    fprintf(stderr, "Error, can't open '%s' for reading.\n", file1);
    fflush(stderr);
    return -1;
  }
  fp2 = fopen(file2, "rb");
  if (!fp2)
  {
    fclose(fp1);
    fprintf(stderr, "Error, can't open '%s' for reading.\n", file2);
    fflush(stderr);
    return -1;
  }
  fread(&M,   sizeof(M),  1, fp1);
  fread(&N,   sizeof(N),  1, fp1);
  fread(&N2,  sizeof(N2), 1, fp2);
  fread(&P,   sizeof(P),  1, fp2);
  if (M < 0 || M != N || M != N2 || M != P)
  {
    fclose(fp1);
    fclose(fp2);
    fprintf(stderr, "Error, matrices must be square with positive matrices.\n");
    fflush(stderr);
    return -1;
  }
  return M;
}

void matrixMultCPU(void * param)
{
  int numGPUs = dcgn::globalGPUCount();
  int subSize = 0, subMatDim;
  switch (numGPUs)
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
    if (dcgn::getRank() == 0)
    {
      fprintf(stderr, "Error, must have a square number of processors.\n");
      fflush(stderr);
    }
    dcgn::abort(dcgn::DCGN_ERROR_ABORTED);
    break;
  }

  if (dcgn::getRank() != 0)
  {
    dcgn::barrier();
    for (int i = 1; i < subSize; ++i) dcgn::barrier();
    dcgn::barrier();
    return;
  }
  char ** files = reinterpret_cast<char ** >(param);
  int M, N, N2, P;
  float * sendData;
  FILE * fp1 = fopen(files[0], "rb");
  FILE * fp2 = fopen(files[1], "rb");
  fread(&M,   sizeof(M),  1, fp1);
  fread(&N,   sizeof(N),  1, fp1);
  fread(&N2,  sizeof(N2), 1, fp2);
  fread(&P,   sizeof(P),  1, fp2);

  subMatDim = M / subSize;

  sendData = reinterpret_cast<float * >(malloc(subMatDim * subMatDim * sizeof(float)));

  for (int i = 0; i < numGPUs; ++i)
  {
    int r = i / subSize;
    int c = r + i % subSize;
    if (c >= subSize) c -= subSize;
    for (int j = 0; j < subMatDim; ++j)
    {
      fseek(fp1, sizeof(int) * 2 + (r + j) * M * sizeof(float) + sizeof(float) * c, SEEK_SET);
      fread(sendData + subMatDim * j, sizeof(float) * subMatDim, 1, fp1);
    }
    dcgn::send(dcgn::getGPUID(i, 0), sendData, subMatDim * subMatDim * sizeof(float));

    c = i % subSize;
    r = c + i / subSize;
    if (r >= subSize) r -= subSize;
    for (int j = 0; j < subMatDim; ++j)
    {
      fseek(fp2, sizeof(int) * 2 + (r + j) * M * sizeof(float) + sizeof(int) * c, SEEK_SET);
      fread(sendData + subMatDim * j, sizeof(float) * subMatDim, 1, fp2);
    }
    dcgn::send(dcgn::getGPUID(i, 0), sendData, subMatDim * subMatDim * sizeof(float));
  }

  fclose(fp1);
  fclose(fp2);
  free(sendData);

  dcgn::barrier();
  double t = wallTime();
  for (int i = 1; i < subSize; ++i) dcgn::barrier();
  dcgn::barrier();
  t = wallTime() - t;
  printf("done, took %.3f seconds.\n", t);
}

__host__ void matrixMultGPU(void * param, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  float * gpuA, * gpuB, * gpuC;
  int dim = *reinterpret_cast<int * >(param);
  int subSize = 0, numGPUs = dcgn::globalGPUCount(), subMatDim;
  volatile int * syncblocksArr;

  switch (numGPUs)
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
    break;
  }
  subMatDim = dim / subSize;

  CHECK_ERR();
  cudaMalloc((void ** )&syncblocksArr, sizeof(int) * gridSize.x);                                                             CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuA), sizeof(float) * subMatDim * subMatDim);                                       CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuB), sizeof(float) * subMatDim * subMatDim);                                       CHECK_ERR();
  cudaMalloc(reinterpret_cast<void ** >(&gpuC), sizeof(float) * subMatDim * subMatDim);                                       CHECK_ERR();
  cudaMemset(gpuC, 0, sizeof(float) * subMatDim * subMatDim);                                                                 CHECK_ERR();
  cudaMemset((void * )syncblocksArr, 0, sizeof(int) * gridSize.x);                                                            CHECK_ERR();
  matrixMult<<<gridSize, blockSize, sharedMemSize, *stream>>>(subMatDim, subSize, gpuA, gpuB, gpuC, libParam, syncblocksArr); CHECK_ERR();
}

__global__ void matrixMult(const int subMatDim, const int subSize, float * gpuA, float * gpuB, float * gpuC, const dcgn::GPUInitRequest libParam, volatile int * sbArr)
{
  dcgn::gpu::init(libParam);
  if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
  {
    dcgn::CommStatus stat;
    dcgn::gpu::recv(0, 0, gpuA, sizeof(float) * subMatDim * subMatDim, &stat);
    dcgn::gpu::recv(0, 0, gpuB, sizeof(float) * subMatDim * subMatDim, &stat);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) { dcgn::gpu::barrier(0); }
  __syncblocks(sbArr);
  for (int i = 0; i < subSize; ++i)
  {
    matrixMultSub(subMatDim, subMatDim, subMatDim, gpuA, gpuB, gpuC);
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && i + 1 != subSize)
    {
      int left, right, up, down;
      {
        int id = dcgn::gpu::getRank(0) - 1;
        const int r   = id / subSize, c = id % subSize;
        left  = dcgn::gpu::getGPUID( (c == 0            ? subSize : c) - 1  + subSize * r, 0);
        right = dcgn::gpu::getGPUID( (c == subSize - 1  ? -1      : c) + 1  + subSize * r, 0);
        up    = dcgn::gpu::getGPUID(((r == 0            ? subSize : r) - 1) * subSize + c, 0);
        down  = dcgn::gpu::getGPUID(((r == subSize - 1  ? -1      : r) + 1) * subSize + c, 0);
      }
      dcgn::CommStatus stat;
      dcgn::gpu::sendRecvReplace(0, left, right, gpuA, subMatDim * subMatDim * sizeof(float), &stat);
      dcgn::gpu::sendRecvReplace(0, up,   down,  gpuB, subMatDim * subMatDim * sizeof(float), &stat);
      dcgn::gpu::barrier(0);
    }
    if (i + 1 != subSize)
    {
      __syncblocks(sbArr);
    }
  }
  __syncblocks(sbArr);
  if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) { dcgn::gpu::barrier(0); }
}

__shared__ float mem[1024];

__device__ void matrixMultSub(const int M, const int N, const int P, const float * const A, const float * const B, float * const C)
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

__device__ void __syncblocks(volatile int * syncblocksArr)
{
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
  {
    syncblocksArr[blockIdx.x] = 1;
    if (blockIdx.x == 0)
    {
      for (int i = 1; i < gridDim.x; ++i)
      {
        while (syncblocksArr[i] == 0) { }
      }
      for (int i = 0; i < gridDim.x; ++i)
      {
        syncblocksArr[i] = 0;
      }
    }
    while (syncblocksArr[blockIdx.x] == 1) { }
  }
  __syncthreads();
}
