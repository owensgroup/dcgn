#include <mpi.h>
#include <cmath>
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
typedef struct _CommPacket
{
  int startRow, endRow;
} CommPacket;

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

void scanRow(const int row, const MandelbrotInfo & minfo, int * pixels)
{
  const float yVal = static_cast<float>(row) / static_cast<float>(minfo.height  - 1);
  for (int p = 0; p < minfo.width; ++p)
  {
    int iter;
    float z, zi, mag;
    const float xVal = static_cast<float>(p) / static_cast<float>(minfo.width - 1);

    z = zi = mag = 0.0f;
    const float x = minfo.xMin + (minfo.xMax - minfo.xMin) * xVal;
    const float y = minfo.yMin + (minfo.yMax - minfo.yMin) * yVal;

    for (iter = 0; mag < 4.0f && iter <= minfo.maxIters; ++iter)
    {
      const float t = z * z - zi * zi + x;
      zi = 2.0f * z * zi + y;
      z = t;
      mag = z * z + zi * zi;
    }
    *(pixels++) = --iter;
  }
}

void scanConvertRows(const int startRow, const int endRow, const MandelbrotInfo & minfo, int * pixels)
{
  for (int i = startRow; i < endRow; ++i)
  {
    scanRow(i, minfo, pixels + (i - startRow) * minfo.width);
  }
}

void convertHSBtoRGB(const float & hue, const float & brightness, unsigned char * pixel)
{
  unsigned char r = 0, g = 0, b = 0;
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

void doSlave()
{
  MandelbrotInfo minfo;
  CommPacket packet;
  MPI_Bcast(&minfo, sizeof(minfo), MPI_BYTE, 0, MPI_COMM_WORLD);
  int * pixels = new int[minfo.width * minfo.maxRows];
  packet.startRow = packet.endRow = -1;
  MPI_Barrier(MPI_COMM_WORLD);
  do
  {
    MPI_Status stat;
    MPI_Send(&packet, sizeof(CommPacket), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    if (packet.startRow < packet.endRow)
    {
      MPI_Send(pixels, sizeof(int) * minfo.width * (packet.endRow - packet.startRow), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Recv(&packet, sizeof(CommPacket), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &stat);
    if (packet.startRow < packet.endRow)
    {
      scanConvertRows(packet.startRow, packet.endRow, minfo, pixels);
    }
  }
  while (packet.startRow < packet.endRow);
  MPI_Barrier(MPI_COMM_WORLD);
  delete [] pixels;
}

void doMaster(const char * const inputFile, const char * const outputFile)
{
  CommPacket packet;
  int size, startOfImage, nextRow = 0;
  MandelbrotInfo minfo;
  readInputFile(inputFile, minfo);
  FILE * outfp = fopen(outputFile, "wb");
  if (!outfp)
  {
    fprintf(stderr, "Error, couldn't open %s for writing.\n", outputFile);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  MPI_Bcast(&minfo, sizeof(minfo), MPI_BYTE, 0, MPI_COMM_WORLD);

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  unsigned char * rgb = new unsigned char[3 * minfo.width * minfo.maxRows];
  int * pixels = new int[minfo.width * minfo.maxRows];
  int numKilled = 1;

  fprintf(outfp, "P6\n%d %d\n255\n%n", minfo.width, minfo.height, &startOfImage);

  double timer = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  while (nextRow < minfo.height)
  {
    MPI_Status stat;
    MPI_Recv(&packet, sizeof(CommPacket), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
    if (packet.endRow > packet.startRow)
    {
      MPI_Recv(pixels, sizeof(int) * minfo.width * (packet.endRow - packet.startRow), MPI_BYTE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
      storeRows(outfp, startOfImage, stat.MPI_SOURCE - 1, size - 1, packet.startRow, packet.endRow, minfo, pixels, rgb);
    }
    packet.startRow = nextRow;
    packet.endRow = std::min(packet.startRow + minfo.maxRows, minfo.height);
    nextRow = packet.endRow;
    MPI_Send(&packet, sizeof(CommPacket), MPI_BYTE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
  }
  while (numKilled < size)
  {
    MPI_Status stat;
    MPI_Recv(&packet, sizeof(CommPacket), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
    if (packet.endRow > packet.startRow)
    {
      MPI_Recv(pixels, sizeof(int) * minfo.width * (packet.endRow - packet.startRow), MPI_BYTE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
      storeRows(outfp, startOfImage, stat.MPI_SOURCE - 1, size - 1, packet.startRow, packet.endRow, minfo, pixels, rgb);
    }
    packet.startRow = packet.endRow = -1;
    ++numKilled;
    MPI_Send(&packet, sizeof(CommPacket), MPI_BYTE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  timer = MPI_Wtime() - timer;
  fprintf(stderr, "done, took %f seconds.\n", timer);

  fclose(outfp);
  delete [] rgb;
  delete [] pixels;
}

int main(int argc, char ** argv)
{
  int rank;
  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    fflush(stderr);
    return 1;
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)  doMaster(argv[1], argv[2]);
  else            doSlave();

  MPI_Finalize();

  return 0;
}
