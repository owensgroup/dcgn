#include <dcgn/dcgn.h>
#include <dcgn/CUDAFunctions.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cerrno>
#include <sched.h>

#define CHECK_ERROR()                                                                               \
{                                                                                                   \
  cudaError_t err = cudaGetLastError();                                                             \
  if (err != cudaSuccess)                                                                           \
  {                                                                                                 \
    fprintf(stderr, "%s.%s.%d: %s.\n", __FILE__, __FUNCTION__, __LINE__, cudaGetErrorString(err));  \
    fflush(stderr);                                                                                 \
    exit(1);                                                                                        \
  }                                                                                                 \
}                                                                                                   \

typedef struct _MandelbrotInfo
{
  int width, height, maxRows, maxIters;
  float xMin, xMax, yMin, yMax;
} MandelbrotInfo;
typedef struct _CommPacket
{
  int startRow, endRow;
} CommPacket;

void convertHSBtoRGB(const float & hue, const float & brightness, unsigned char * pixel)
{
  unsigned char r, g, b;
  const float saturation = 1.0f;
  float h = (hue - floor(hue)) * 6.0f;
  float f = h - floor(h);
  float p = brightness * (1.0f - saturation);
  float q = brightness * (1.0f - saturation * f);
  float t = brightness * (1.0f - (saturation * (1.0f - f)));
  switch (static_cast<int>(h))
  {
  case 0:
    r = static_cast<unsigned char>(brightness * 255.0f + 0.5f);
    g = static_cast<unsigned char>(t * 255.0f + 0.5f);
    b = static_cast<unsigned char>(p * 255.0f + 0.5f);
    break;
  case 1:
    r = static_cast<unsigned char>(q * 255.0f + 0.5f);
    g = static_cast<unsigned char>(brightness * 255.0f + 0.5f);
    b = static_cast<unsigned char>(p * 255.0f + 0.5f);
    break;
  case 2:
    r = static_cast<unsigned char>(p * 255.0f + 0.5f);
    g = static_cast<unsigned char>(brightness * 255.0f + 0.5f);
    b = static_cast<unsigned char>(t * 255.0f + 0.5f);
    break;
  case 3:
    r = static_cast<unsigned char>(p * 255.0f + 0.5f);
    g = static_cast<unsigned char>(q * 255.0f + 0.5f);
    b = static_cast<unsigned char>(brightness * 255.0f + 0.5f);
    break;
  case 4:
    r = static_cast<unsigned char>(t * 255.0f + 0.5f);
    g = static_cast<unsigned char>(p * 255.0f + 0.5f);
    b = static_cast<unsigned char>(brightness * 255.0f + 0.5f);
    break;
  case 5:
    r = static_cast<unsigned char>(brightness * 255.0f + 0.5f);
    g = static_cast<unsigned char>(p * 255.0f + 0.5f);
    b = static_cast<unsigned char>(q * 255.0f + 0.5f);
    break;
  }
  pixel[0] = r;
  pixel[1] = g;
  pixel[2] = b;
}

void readInputFile(const char * const input, MandelbrotInfo & minfo)
{
  FILE * fp = fopen(input, "r");
  char line[2048];

  if (!fp)
  {
    fprintf(stderr, "Error, couldn't open file '%s' for reading.\n", input);
    fflush(stderr);
    exit(1);
  }

  while (fgets(line, 2047, fp))
  {
    char * ptr = line;
    while (*ptr && *ptr <= ' ') ++ptr;
    if (*ptr == '#') continue;
    char * end = ptr + strlen(ptr) - 1;
    while (end >= ptr && *end <= ' ') --end;
    *(end + 1) = 0;

    char var[1024];
    int ival;
    float fval;
    sscanf(ptr, "%s = %d", var, &ival);
    sscanf(ptr, "%s = %f", var, &fval);
    if      (strcmp(var, "width")     == 0) minfo.width    = ival;
    else if (strcmp(var, "height")    == 0) minfo.height   = ival;
    else if (strcmp(var, "maxRows")   == 0) minfo.maxRows  = ival;
    else if (strcmp(var, "maxIters")  == 0) minfo.maxIters = ival;
    else if (strcmp(var, "xmin")      == 0) minfo.xMin     = fval;
    else if (strcmp(var, "xmax")      == 0) minfo.xMax     = fval;
    else if (strcmp(var, "ymin")      == 0) minfo.yMin     = fval;
    else if (strcmp(var, "ymax")      == 0) minfo.yMax     = fval;
    else
    {
      fprintf(stderr, "Warning, skipping invalid variable in input file (%s).\n", var);
      fflush(stderr);
    }
  }

  fclose(fp);
}

void storeRows(FILE * outfp, const int startOfImage, const int source, const int size,
               const int startRow, const int endRow, const MandelbrotInfo & minfo,
               const int * const pixels, unsigned char * const rgb)
{
  const float hue = static_cast<float>(source) / static_cast<float>(size);
  unsigned char * pixel = rgb;
  for (int row = startRow; row < endRow; ++row)
  {
    const int * rowp = pixels + (row - startRow) * minfo.width;
    for (int i = 0; i < minfo.width; ++i)
    {
      float t = 0.0f;
      if (rowp[i] == 0)
      {
        t = 0.0f;
      }
      else if (rowp[i] < 16)
      {
        t = 0.75f * (static_cast<float>(rowp[i]) - 1.0f) / 14.0f;
      }
      else
      {
        t = 0.75f + 0.25f * (static_cast<float>(rowp[i]) - 16.0f) / static_cast<float>(minfo.maxIters - 16);
      }
      convertHSBtoRGB(hue, t, pixel);
      pixel += 3;
    }
  }

  fseek(outfp, startOfImage + sizeof(unsigned char) * minfo.width * startRow * 3, SEEK_SET);
  fwrite(rgb, sizeof(unsigned char) * minfo.width * (endRow - startRow) * 3, 1, outfp);
}

__device__ void scanRow(const MandelbrotInfo minfo, const int row, int * pixels)
{
  const float dx = minfo.xMax - minfo.xMin;
  const float dy = minfo.yMax - minfo.yMin;
  const float yVal = static_cast<float>(row) / static_cast<float>(minfo.height  - 1);

  for (int p = threadIdx.x; p < minfo.width; p += blockDim.x)
  {
    int iter = 0;
    float z, zi, mag;
    const float xVal = static_cast<float>(p) / static_cast<float>(minfo.width - 1);

    z = zi = mag = 0.0f;
    const float x = minfo.xMin + dx * xVal;
    const float y = minfo.yMin + dy * yVal;

    for (iter = 0; mag < 4.0f && iter <= minfo.maxIters; ++iter)
    {
      const float t = z * z - zi * zi + x;
      zi = 2.0f * z * zi + y;
      z = t;
      mag = z * z + zi * zi;
    }
    pixels[p] = --iter;
  }
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

__global__ void doGPUSlave(int * pixels, MandelbrotInfo * pMinfo, CommPacket * packet, int * sbarr, const dcgn::GPUInitRequest libParam)
{
  dcgn::gpu::init(libParam);

  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    dcgn::gpu::broadcast(0, 0, pMinfo, sizeof(MandelbrotInfo));
  }
  __syncblocks(sbarr);

  MandelbrotInfo minfo = *pMinfo;
  packet->startRow = packet->endRow = -1;
  if (blockIdx.x == 0 && threadIdx.x == 0) dcgn::gpu::barrier(0);
  do
  {
    dcgn::CommStatus stat;
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      dcgn::gpu::send(0, 0, packet, sizeof(packet));
      if (packet->startRow < packet->endRow)
      {
        dcgn::gpu::send(0, 0, pixels, sizeof(int) * minfo.width * (packet->endRow - packet->startRow));
      }
      dcgn::gpu::recv(0, 0, packet, sizeof(packet), &stat);
    }
    __syncblocks(sbarr); // wait for packet to arrive.
    const int startRow = packet->startRow, endRow = packet->endRow;
    int row = startRow + blockIdx.x;
    while (row < endRow)
    {
      scanRow(minfo, row, pixels + minfo.width * (row - startRow));
      row += gridDim.x;
    }
    __syncblocks(sbarr); // wait for work to finish, cause when we loop, we send info back.
  }
  while (packet->startRow < packet->endRow);
  if (blockIdx.x == 0 && threadIdx.x == 0) dcgn::gpu::barrier(0);
}

__host__ void doSlave(void * dbgInfo, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream)
{
  int * pixels, * sbarr;
  MandelbrotInfo * minfo;
  CommPacket * packet;
  cudaMalloc(reinterpret_cast<void ** >(&sbarr),  sizeof(int) * gridSize.x);    CHECK_ERROR();
  cudaMalloc(reinterpret_cast<void ** >(&packet), sizeof(CommPacket));          CHECK_ERROR();
  cudaMalloc(reinterpret_cast<void ** >(&pixels), sizeof(int) * 1048576 * 100); CHECK_ERROR();
  cudaMalloc(reinterpret_cast<void ** >(&minfo),  sizeof(MandelbrotInfo));      CHECK_ERROR();
  cudaMemset(sbarr, 0, sizeof(int) * gridSize.x);                               CHECK_ERROR();
  cudaMemset(minfo, 0, sizeof(MandelbrotInfo));                                 CHECK_ERROR();
  doGPUSlave<<<gridSize, blockSize, sharedMemSize, *stream>>>(pixels, minfo, packet, sbarr, libParam); CHECK_ERROR();
}

__host__ void gpuDtor(void * dbgInfo)
{
}

__host__ void doMaster(void * arg)
{
  CommPacket packet;
  int size, startOfImage, nextRow = 0;
  MandelbrotInfo minfo;
  char ** argv = reinterpret_cast<char ** >(arg);
  readInputFile(argv[1], minfo);
  FILE * outfp = fopen(argv[2], "wb");
  if (!outfp)
  {
    fprintf(stderr, "Error, couldn't open %s for writing.\n", argv[2]);
    fflush(stderr);
    dcgn::abort(dcgn::DCGN_ERROR_ABORTED);
  }
  dcgn::broadcast(0, &minfo, sizeof(minfo));
  size = dcgn::getSize();

  unsigned char * rgb = new unsigned char[3 * minfo.width * minfo.maxRows];
  int * pixels = new int[minfo.width * minfo.maxRows];
  int numKilled = 1;

  fprintf(outfp, "P6\n%d %d\n255\n%n", minfo.width, minfo.height, &startOfImage);

  dcgn::barrier();
  double timer = dcgn::wallTime();

  int lastRow = -10000;
  while (nextRow < minfo.height)
  {
    if (nextRow - lastRow >= 1000)
    {
      printf("%10d / %10d\r", nextRow, minfo.height); fflush(stdout);
      lastRow = nextRow;
    }
    dcgn::CommStatus stat;
    dcgn::recv(dcgn::ANY_SOURCE, &packet, sizeof(packet), &stat);
    if (packet.endRow > packet.startRow)
    {
      dcgn::recv(stat.src, pixels, sizeof(int) * minfo.width * (packet.endRow - packet.startRow), &stat);
      storeRows(outfp, startOfImage, stat.src - 1, size - 1, packet.startRow, packet.endRow, minfo, pixels, rgb);
    }
    packet.startRow = nextRow;
    packet.endRow = std::min(packet.startRow + minfo.maxRows, minfo.height);
    nextRow = packet.endRow;
    dcgn::send(stat.src, &packet, sizeof(packet));
  }
  printf("                                        \r");
  while (numKilled < size)
  {
    printf("%3d / %3d\r", numKilled, size); fflush(stdout);
    dcgn::CommStatus stat;
    dcgn::recv(dcgn::ANY_SOURCE, &packet, sizeof(packet), &stat);
    if (packet.endRow > packet.startRow)
    {
      dcgn::recv(stat.src, pixels, sizeof(int) * minfo.width * (packet.endRow - packet.startRow), &stat);
      storeRows(outfp, startOfImage, stat.src - 1, size - 1, packet.startRow, packet.endRow, minfo, pixels, rgb);
    }
    packet.startRow = packet.endRow = -1;
    ++numKilled;
    dcgn::send(stat.src, &packet, sizeof(packet));
  }
  printf("                           \r"); fflush(stdout);

  dcgn::barrier();
  timer = dcgn::wallTime() - timer;
  fprintf(stderr, "done, took %f seconds.\n", timer);

  fclose(outfp);
  delete [] rgb;
  delete [] pixels;
}

int main(int argc, char ** argv)
{
  int gpus[] = { 0, 1, -1 };
  uint3 gs = { 12, 1, 1 }, bs = { 160, 1, 1 };
  dcgn::init(&argc, &argv);
  dcgn::initComm(-1);
  dcgn::initGPU(gpus, 1, 0);
  dcgn::initCPU(dcgn::getNodeID() == 0 ? 1 : 0);
  dcgn::start();

  if (argc != 3)
  {
    if (dcgn::getNodeID() == 0)
    {
      fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
      fflush(stderr);
    }
    dcgn::finalize();
    return 1;
  }

  void * gpuMem1, * gpuMem2;
  if (dcgn::getNodeID() == 0) dcgn::launchCPUKernel(0, doMaster, argv);
  dcgn::launchGPUKernel(0, doSlave, gpuDtor, &gpuMem1, gs, bs);
  dcgn::launchGPUKernel(1, doSlave, gpuDtor, &gpuMem2, gs, bs);

  dcgn::finalize();

  return 0;
}
