#include <dcgn/dcgn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cerrno>
#include <sched.h>

typedef struct _MandelbrotInfo
{
  int width, height, maxRows, maxIters;
  float xMin, xMax, yMin, yMax;
} MandelbrotInfo;
typedef struct _CommInfo
{
  int terminate;
  int startRow, endRow;
} CommInfo;
typedef struct _CPUKernelInfo
{
  MandelbrotInfo minfo;
  char * input;
  char * output;
} CPUKernelInfo;

void gpuKernelWrapper(void * number, const dcgn::GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream);

void readInputFile(CPUKernelInfo * const kernelInfo)
{
  FILE * fp = fopen(kernelInfo->input, "r");
  char line[2048];

  if (!fp)
  {
    fprintf(stderr, "Error, couldn't open file '%s' for reading.\n", kernelInfo->input);
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
    if      (strcmp(var, "width")     == 0) kernelInfo->minfo.width    = ival;
    else if (strcmp(var, "height")    == 0) kernelInfo->minfo.height   = ival;
    else if (strcmp(var, "maxRows")   == 0) kernelInfo->minfo.maxRows  = ival;
    else if (strcmp(var, "maxIters")  == 0) kernelInfo->minfo.maxIters = ival;
    else if (strcmp(var, "xMin")      == 0) kernelInfo->minfo.xMin     = fval;
    else if (strcmp(var, "xMax")      == 0) kernelInfo->minfo.xMax     = fval;
    else if (strcmp(var, "yMin")      == 0) kernelInfo->minfo.yMin     = fval;
    else if (strcmp(var, "ymax")      == 0) kernelInfo->minfo.yMax     = fval;
    else
    {
      fprintf(stderr, "Warning, skipping invalid variable in input file (%s).\n", var);
      fflush(stderr);
    }
  }

  fclose(fp);
}

int recvRows(CPUKernelInfo * const kernelInfo,
             FILE * fp,
             const int startOfImage,
             std::vector<float> & pixels,
             std::vector<unsigned char> & interp)
{
  CommInfo info;
  dcgn::CommStatus stat;

  dcgn::recv(dcgn::ANY_SOURCE, &info, sizeof(info), &stat);
  const int numPixels = (info.endRow - info.startRow) * kernelInfo->minfo.width;
  if (static_cast<int>(pixels.size()) < numPixels)
  {
    pixels.resize(numPixels);
    interp.resize(numPixels);
  }
  dcgn::recv(stat.src, &pixels[0], numPixels * sizeof(float), &stat);
  for (int i = 0; i < (int)pixels.size(); ++i)
  {
    interp[i] = static_cast<unsigned char>(pixels[i] * 255.0f);
  }
  fseek(fp, startOfImage + info.startRow * kernelInfo->minfo.width, SEEK_SET);
  fwrite(&interp[0], numPixels, 1, fp);
  fflush(fp);

  return stat.src;
}

void doMaster(const int size, CPUKernelInfo * const kernelInfo)
{
  int nextRow = 0;
  int numKilled = 0;
  int startOfImage = 0;
  FILE * fp = fopen(kernelInfo->output, "wb+");
  std::vector<bool> idle, killed;
  std::vector<float> pixels;
  std::vector<unsigned char> interp;

  if (!fp)
  {
    fprintf(stderr, "Error, couldn't open file '%s' for writing. %s.\n", kernelInfo->output, strerror(errno));
    fflush(stderr);
    dcgn::abort(dcgn::DCGN_ERROR_ABORTED);
  }
  fprintf(fp, "P5\n%d %d\n256\n%n", kernelInfo->minfo.width, kernelInfo->minfo.height, &startOfImage);

  dcgn::broadcast(0, &kernelInfo->minfo, sizeof(kernelInfo->minfo));

  // fprintf(stderr, "master : received mandelbrotInfo { w=%d, h=%d, mr=%d, mi=%d, x={%f,%f}, y={%f,%f} }\n",
  //                 kernelInfo->minfo.width, kernelInfo->minfo.height, kernelInfo->minfo.maxRows, kernelInfo->minfo.maxIters,
  //                 kernelInfo->minfo.xMin,  kernelInfo->minfo.xMax,   kernelInfo->minfo.yMin,    kernelInfo->minfo.yMax); fflush(stderr);

  idle.resize(size, true);
  killed.resize(size, false);

  while (nextRow < kernelInfo->minfo.height)
  {
    int endRow = std::min(nextRow + kernelInfo->minfo.maxRows, kernelInfo->minfo.height);

    int idleIndex = -1;
    for (int i = 1; i < (int)idle.size(); ++i)
    {
      if (idle[i])
      {
        // fprintf(stderr, "%d is idle, sending work.\n", i); fflush(stderr);
        CommInfo info = { 0, nextRow, endRow };
        nextRow = endRow;
        dcgn::send(i, &info, sizeof(info));
        idle[i] = false;
        idleIndex = i;
        break;
      }
    }
    if (idleIndex == -1)
    {
      // fprintf(stderr, "no slave is idle, recving rows.\n"); fflush(stderr);
      int slave = recvRows(kernelInfo, fp, startOfImage, pixels, interp);
      fprintf(stderr, "recvd rows from %d. setting slave to idle.\n", slave); fflush(stderr);
      idle[slave] = true;
    }
  }

  numKilled = 1;

  for (int i = 1; i < (int)idle.size(); ++i)
  {
    if (idle[i])
    {
      // fprintf(stderr, "slave %d is idle, sending terminate command.\n", i); fflush(stderr);
      CommInfo info;
      info.terminate = 1;
      dcgn::send(i, &info, sizeof(info));

      // fprintf(stderr, "marking slave %d as killed.\n", i); fflush(stderr);
      idle[i] = true;
      killed[i] = true;
      ++numKilled;
    }
  }

  while (numKilled < size)
  {
    CommInfo info;

    int slave = recvRows(kernelInfo, fp, startOfImage, pixels, interp);

    fprintf(stderr, "recvd rows from %d. sending terminate command.\n", slave); fflush(stderr);
    info.terminate = 1;
    dcgn::send(slave, &info, sizeof(info));

    // fprintf(stderr, "marking slave %d as killed.\n", slave); fflush(stderr);

    idle[slave] = true;
    killed[slave] = true;
    ++numKilled;
  }

  fclose(fp);

  // fprintf(stderr, "all done, shutting down.\n"); fflush(stderr);

}

void doSlave()
{
  MandelbrotInfo minfo;

  dcgn::broadcast(0, &minfo, sizeof(minfo));
  std::vector<float> pixels;

  // fprintf(stderr, "slave %d: received mandelbrotInfo { w=%d, h=%d, mr=%d, mi=%d, x={%f,%f}, y={%f,%f} }\n",
  //                 static_cast<int>(dcgn::getRank()),
  //                 minfo.width, minfo.height, minfo.maxRows, minfo.maxIters,
  //                 minfo.xMin, minfo.xMax, minfo.yMin, minfo.yMax); fflush(stderr);

  pixels.resize(minfo.maxRows * minfo.width);

  CommInfo info;
  dcgn::CommStatus stat;
  do
  {
    // fprintf(stderr, "slave %d: recving command from master.\n", static_cast<int>(dcgn::getRank())); fflush(stderr);

    dcgn::recv(0, &info, sizeof(info), &stat);

    // fprintf(stderr, "slave %d: info.terminate? '%s'.\n", static_cast<int>(dcgn::getRank()), info.terminate ? "true" : "false"); fflush(stderr);

    if (!info.terminate)
    {
      int pixelIndex = 0;
      for (int row = info.startRow; row < info.endRow; ++row)
      {
        for (int p = 0; p < minfo.width; ++p)
        {
          int iter;
          float z, zi, mag;
          float x = static_cast<float>(p) / static_cast<float>(minfo.width - 1);
          float y = static_cast<float>(row) / static_cast<float>(minfo.height  - 1);

          z = zi = mag = 0.0f;
          x = minfo.xMin + (minfo.xMax - minfo.xMin) * x;
          y = minfo.yMin + (minfo.yMax - minfo.yMin) * y;

          for (iter = 0; mag < 4.0f && iter < minfo.maxIters; ++iter)
          {
            const float t = z * z - zi * zi + x;
            zi = 2 * z * zi + y;
            z = t;
            mag = z * z + zi * zi;
          }
          pixels[++pixelIndex] = static_cast<float>(iter) / static_cast<float>(minfo.maxIters);
        }
      }
      // fprintf(stderr, "slave %d: sending header back to master.\n", static_cast<int>(dcgn::getRank())); fflush(stderr);
      dcgn::send(0, &info, sizeof(info));
      // fprintf(stderr, "slave %d: sending pixels to master.\n", static_cast<int>(dcgn::getRank())); fflush(stderr);
      dcgn::send(0, &pixels[0], sizeof(float) * minfo.width * (info.endRow - info.startRow));
    }
  }
  while (!info.terminate);

  // fprintf(stderr, "all done, slave %d exiting.\n", static_cast<int>(dcgn::getRank())); fflush(stderr);
}

void cpuKernel(void * cpuKernelInfo)
{
  if (dcgn::getRank() == 0)
  {
    doMaster(dcgn::getSize(), reinterpret_cast<CPUKernelInfo * >(cpuKernelInfo));
  }
  else
  {
    doSlave();
  }
}

int main(int argc, char ** argv)
{
  // int cpus[] = { 0 };
  int gpus[] = { 0, 1, -1 };
  uint3 gridSize  = { 16, 1, 1 };
  uint3 blockSize = { 256, 1, 1 };

  dcgn::init(&argc, &argv, 2, gpus, 1, 0, -1);

  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    fflush(stderr);
    dcgn::finalize();
    return 0;
  }
  CPUKernelInfo kernelInfo;
  kernelInfo.input = argv[1];
  kernelInfo.output = argv[2];
  readInputFile(&kernelInfo);

  dcgn::launchCPUKernel(0, cpuKernel,        reinterpret_cast<void * >(&kernelInfo));
  dcgn::launchCPUKernel(1, cpuKernel,        reinterpret_cast<void * >(&kernelInfo));
  dcgn::launchGPUKernel(0, gpuKernelWrapper, 0, reinterpret_cast<void * >(&kernelInfo.minfo), gridSize, blockSize, 0);
  dcgn::launchGPUKernel(1, gpuKernelWrapper, 0, reinterpret_cast<void * >(&kernelInfo.minfo), gridSize, blockSize, 0);

  while (!dcgn::areAllLocalResourcesIdle())
  {
    sched_yield();
  }

  dcgn::finalize();

  return 0;
}
