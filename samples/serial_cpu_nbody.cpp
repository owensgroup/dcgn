#ifndef __host__
  #define __host__
#endif
#ifndef __device__
  #define __device__
#endif
#ifndef __inline__
  #define __inline__
#endif

#include "samples/Body.cxx"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <sys/time.h>

double getTime()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1000000.0;
}

float getRandom(const float low = 0.0f, const float high = 1.0f)
{
#ifdef _WIN32
  return (rand() / (double)RAND_MAX) * (high - low) + low;
#else
  return static_cast<float>(drand48() * (high - low) + low);
#endif
}
void seedRandom(const int val = 0)
{
#ifdef _WIN32
  srand(val);
#else
  srand48(val);
#endif
}

void genFile(int numBodies, char * outputFile)
{
  FILE * fp = fopen(outputFile, "wb");

  seedRandom(time(0));

  Body * bodies = new Body[numBodies];

  for (int i = 0; i < numBodies; ++i)
  {
    Body & b = bodies[i];
    const float theta1  = getRandom(0.0f, 6.28318f);
    const float phi1    = getRandom(0.0f, 3.14159f);
    const float radius1 = getRandom(0.0f, 1000000000.0f);
    const float theta2  = getRandom(0.0f, 6.28318f);
    const float phi2    = getRandom(0.0f, 3.14159f);
    const float radius2 = getRandom(0.0f, 10.0f);
    b.x     = radius1 * sinf(theta1) * cosf(phi1);
    b.y     = radius1 * sinf(theta1) * sinf(phi1);
    b.z     = radius1 * cosf(theta1);
    b.vx    = radius2 * sinf(theta2) * cosf(phi2);
    b.vy    = radius2 * sinf(theta2) * sinf(phi2);
    b.vz    = radius2 * cosf(theta2);
    b.mass  = getRandom(1e12, 1e15);
  }
  fwrite(&numBodies, sizeof(int), 1, fp);
  int t = 1; fwrite(&t, sizeof(int), 1, fp);
  fwrite(bodies, sizeof(Body) * numBodies, 1, fp);
  fclose(fp);
}

int main(int argc, char ** argv)
{
  Body * bodies, * updatedBodies;
  int size, timeSteps;
  double timeDelta;
  FILE * fp, * outfp;

  if (argc == 3)
  {
    int nb;
    sscanf(argv[1], "%d", &nb);
    genFile(nb, argv[2]);
    return 0;
  }
  if (argc != 5)
  {
    fprintf(stderr, "Usage: %s <input_file> <output_file> <time_steps> <time_delta>\n", argv[0]);
    return 1;
  }
  fp = fopen(argv[1], "rb");
  if (!fp)
  {
    fprintf(stderr, "Couldn't open %s for reading\n", argv[1]);
    return 1;
  }
  outfp = fopen(argv[2], "wb");
  if (!outfp)
  {
    fclose(fp);
    fprintf(stderr, "Couldn't open %s for writing\n", argv[2]);
    return 1;
  }

  sscanf(argv[3], "%d", &timeSteps);
  sscanf(argv[4], "%lf", &timeDelta);

  int dummy;

  fread(&size, sizeof(int), 1, fp);
  fread(&dummy, sizeof(int), 1, fp);
  bodies = new Body[size];
  updatedBodies = new Body[size];
  fread(bodies, sizeof(Body) * size, 1, fp);
  fclose(fp);

  fwrite(&size,       sizeof(int), 1, outfp);
  fwrite(&timeSteps,  sizeof(int), 1, outfp);

  double start = getTime();

  for (int i = 0; i < timeSteps; ++i)
  {
    memcpy(updatedBodies, bodies, sizeof(Body) * size);
    for (int j = 0; j < size; ++j)
    {
      updatedBodies[j] = bodies[j];
      for (int k = 0; k < size; ++k)
      {
        if (j != k)
        {
          updatedBodies[j].addForceFrom(bodies[k]);
        }
      }
      updatedBodies[j].update(timeDelta);
    }
    fwrite(updatedBodies, sizeof(Body) * size, 1, outfp);
    if (i + 1 < timeSteps) memcpy(bodies, updatedBodies, sizeof(Body) * size);
  }
  fclose(outfp);

  double time = getTime() - start;

  printf("done, took %.3fs.\n", time);

  delete [] bodies;
  delete [] updatedBodies;

  return 0;
}
