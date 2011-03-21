#ifndef __BODY_CXX__
#define __BODY_CXX__

#include <cstdio>
#include <cmath>

class Body
{
  public:
    float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
    float mass;

    __host__ __device__ Body() : x(0), y(0), z(0), vx(0), vy(0), vz(0), ax(0), ay(0), az(0), mass(0)
    {
    }
    __host__ __device__ inline Body(const Body & rhs)
    {
      *this = rhs;
    }
    __host__ __device__ inline Body & operator = (const Body & rhs)
    {
      if (this != &rhs)
      {
        // memcpy(this, &rhs, sizeof(*this));
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        vx = rhs.vx;
        vy = rhs.vy;
        vz = rhs.vz;
        ax = rhs.ax;
        ay = rhs.ay;
        az = rhs.az;
        mass = rhs.mass;
      }
      return * this;
    }
    __host__ __device__ void addForceFrom(const Body & rhs)
    {
      const float G = 6.67300e-11;
      if (x == rhs.x && y == rhs.y && z == rhs.z) return;
      float dx = rhs.x - x, dy = rhs.y - y, dz = rhs.z - z;
      float r = sqrt(dx * dx + dy * dy + dz * dz);
      float rSquared = r * r;
      float t = G * mass * rhs.mass / rSquared;
      ax += (t * (dx / r)) / mass;
      ay += (t * (dy / r)) / mass;
      az += (t * (dz / r)) / mass;
    }
    __host__ __device__ void update(const double timeDelta)
    {
      vx += ax * timeDelta;
      vy += ay * timeDelta;
      vz += az * timeDelta;
      x += vx * timeDelta;
      y += vy * timeDelta;
      z += vz * timeDelta;
      ax = ay = az = 0;
    }
};

#endif
