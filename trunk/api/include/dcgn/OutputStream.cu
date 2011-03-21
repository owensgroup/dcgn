#ifndef __DCGN_OUTPUTSTREAM_CU__
#define __DCGN_OUTPUTSTREAM_CU__

#ifdef __CUDA__

#include <dcgn/dcgn.h>
#include <dcgn/OutputStream.h>

namespace dcgn
{
  __device__ OutputStream & OutputStream::format(const char * const s)
  {
    int len;
    ptr = reinterpret_cast<char * >(memory);
    fmt = s;
    for (len = 0; s[len]; ++len) { }
    fmtLen = len;
    return * this;
  }
  __device__ OutputStream & OutputStream::arg(const          char          p)
  {
    return arg(static_cast<int>(p));
  }
  __device__ OutputStream & OutputStream::arg(const unsigned char          p)
  {
    return arg(static_cast<unsigned int>(p));
  }
  __device__ OutputStream & OutputStream::arg(const          short         p)
  {
    return arg(static_cast<int>(p));
  }
  __device__ OutputStream & OutputStream::arg(const unsigned short         p)
  {
    return arg(static_cast<unsigned int>(p));
  }
  __device__ OutputStream & OutputStream::arg(const          int           p)
  {
    *reinterpret_cast<int * >(ptr) = p;
    ptr += sizeof(int);
    return * this;
  }
  __device__ OutputStream & OutputStream::arg(const unsigned int           p)
  {
    *reinterpret_cast<unsigned int * >(ptr) = p;
    ptr += sizeof(unsigned int);
    return * this;
  }
  __device__ OutputStream & OutputStream::arg(const          long long     p)
  {
    *reinterpret_cast<long long * >(ptr) = p;
    ptr += sizeof(long long);
    return * this;
  }
  __device__ OutputStream & OutputStream::arg(const unsigned long long     p)
  {
    *reinterpret_cast<unsigned long long * >(ptr) = p;
    ptr += sizeof(unsigned long long);
    return * this;
  }
  __device__ OutputStream & OutputStream::arg(const float                  p)
  {
    return arg(static_cast<double>(p));
  }
  __device__ OutputStream & OutputStream::arg(const double                 p)
  {
    *reinterpret_cast<double * >(ptr) = p;
    ptr += sizeof(double);
    return * this;
  }
  __device__ OutputStream & OutputStream::arg(const          char * const  p)
  {
    *reinterpret_cast<const char ** >(ptr) = p;
    ptr += sizeof(const char * );
    return * this;
  }
  __device__ OutputStream & OutputStream::arg(const unsigned char * const  p)
  {
    *reinterpret_cast<const unsigned char ** >(ptr) = p;
    ptr += sizeof(const unsigned char * );
    return * this;
  }
  __device__ OutputStream & OutputStream::arg(const          void * const  p)
  {
    *reinterpret_cast<const void ** >(ptr) = p;
    ptr += sizeof(const void * );
    return * this;
  }
  __device__ OutputStream & OutputStream::args(const void * const arguments, const int len)
  {
    const char * p1 = reinterpret_cast<const char * >(arguments);
    const char * p2 = p1 + len;
    while (p2 - p1 >= 4)
    {
      *reinterpret_cast<int * >(ptr) = *reinterpret_cast<const int * >(p1);
      ptr += sizeof(int);
      p1 += sizeof(int);
    }
    while (p1 != p2)
    {
      *(ptr++) = *(p1++);
    }
    return * this;
  }
  __device__ OutputStream & OutputStream::finish()
  {
    return flush();
  }

  __device__ OutputStream & OutputStream::flush()
  {
    memLen = ptr - reinterpret_cast<char * >(memory);
    *cpuFlag = 1;
    while (*cpuFlag == 1) { }
    ptr = reinterpret_cast<char * >(memory);
    return * this;
  }
}

#endif

__device__ dcgn::OutputStream * dcgn_gpu_gpuOutputStream;

#endif
