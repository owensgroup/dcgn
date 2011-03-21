#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cerrno>
#include <sys/time.h>

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

double wallTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1000000.0;
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

__global__ void scanRow(const MandelbrotInfo minfo, int * pPixels)
{
  const int row = blockIdx.x;
  const float dx = minfo.xMax - minfo.xMin;
  const float dy = minfo.yMax - minfo.yMin;
  const float yVal = static_cast<float>(row) / static_cast<float>(minfo.height  - 1);

  int * pixels = pPixels + row * minfo.width;
  for (int p = threadIdx.x; p < minfo.width; p += blockDim.x)
  {
    int iter;
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

void storeRows(FILE * outfp, const int startOfImage,
               const int startRow, const int endRow, const MandelbrotInfo & minfo,
               const int * const pixels, unsigned char * const rgb)
{
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
      *(pixel++) = static_cast<unsigned char>(t * 255.0f);
      *(pixel++) = 0;
      *(pixel++) = 0;
    }
  }

  fseek(outfp, startOfImage + sizeof(unsigned char) * minfo.width * startRow * 3, SEEK_SET);
  fwrite(rgb, sizeof(unsigned char) * minfo.width * (endRow - startRow) * 3, 1, outfp);
}

__global__ void scanRow(const MandelbrotInfo minfo, const int startRow, int * pPixels)
{
  const int row = blockIdx.x + startRow;
  const float dx = minfo.xMax - minfo.xMin;
  const float dy = minfo.yMax - minfo.yMin;
  const float yVal = static_cast<float>(row) / static_cast<float>(minfo.height  - 1);

  int * pixels = pPixels + blockIdx.x * minfo.width;
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

__host__ int main(int argc, char ** argv)
{
  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    fflush(stderr);
    return 1;
  }
  int startOfImage;
  MandelbrotInfo minfo;
  readInputFile(argv[1], minfo);
  FILE * outfp = fopen(argv[2], "wb");
  if (!outfp)
  {
    fprintf(stderr, "Error, couldn't open %s for writing.\n", argv[2]);
    fflush(stderr);
    return 1;
  }
  unsigned char * rgb = new unsigned char[3 * minfo.width * minfo.maxRows];
  int * pixels, * gpuPixels;
  double t = 0.0, t0 = 0.0, t1 = 0.0, t2;

  cudaMalloc    (reinterpret_cast<void ** >(&gpuPixels), sizeof(int) * minfo.width * minfo.maxRows); CHECK_ERROR();
  cudaMallocHost(reinterpret_cast<void ** >(&pixels),    sizeof(int) * minfo.width * minfo.maxRows); CHECK_ERROR();
  fprintf(outfp, "P6\n%d %d\n255\n%n", minfo.width, minfo.height, &startOfImage);
  t = wallTime();
  int row = 0;
  while (row < minfo.height)
  {
    uint3 gs = { std::min(row + minfo.maxRows, minfo.height) - row, 1, 1 };
    uint3 bs = { 256, 1, 1 };
    t2 = wallTime();
    scanRow<<<gs, bs>>>(minfo, row, gpuPixels);                                                       CHECK_ERROR();
    cudaThreadSynchronize();                                                                          CHECK_ERROR();
    t2 = wallTime() - t2;
    t0 += t2;
    t2 = wallTime();
    cudaMemcpy(pixels, gpuPixels, sizeof(int) * minfo.width * minfo.maxRows, cudaMemcpyDeviceToHost);  CHECK_ERROR();
    t2 = wallTime() - t2;
    t1 += t2;
    storeRows(outfp, startOfImage, row, gs.x, minfo, pixels, rgb);
    row += minfo.maxRows;
  }
  t = wallTime() - t;
  fclose(outfp);
  cudaFree(gpuPixels);
  cudaFreeHost(pixels);
  delete [] rgb;

  // printf("took %f seconds in kernel, %f seconds in memcpy.\n", t0, t1);
  printf("done, took %f seconds.\n", t);

  return 0;
}
