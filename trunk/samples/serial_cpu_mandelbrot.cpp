#include <dcgn/dcgn.h>
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

int main(int argc, char ** argv)
{
  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    fflush(stderr);
    return 1;
  }
  MandelbrotInfo minfo;
  readInputFile(argv[1], minfo);
  FILE * outfp = fopen(argv[2], "wb");
  if (!outfp)
  {
    fprintf(stderr, "Error, couldn't open %s for writing.\n", argv[2]);
    fflush(stderr);
    return 1;
  }
  unsigned char * rgb = new unsigned char[3 * minfo.width * minfo.height];
  int * pixels = new int[minfo.width * minfo.height];
  unsigned char * pixel = rgb;

  fprintf(outfp, "P6\n%d %d\n255\n", minfo.width, minfo.height);
  for (int row = 0; row < minfo.height; ++row)
  {
    int * rowp = pixels + row * minfo.width;
    scanRow(row, minfo, pixels + row * minfo.width);
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
  fwrite(rgb, sizeof(unsigned char) * minfo.width * minfo.height * 3, 1, outfp);
  fclose(outfp);
  delete [] rgb;
  delete [] pixels;

  return 0;
}
