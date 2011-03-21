#ifndef __DCGN_OUTPUTSTREAM_H__
#define __DCGN_OUTPUTSTREAM_H__

#include <dcgn/dcgn.h>
#include <dcgn/Request.h>

namespace dcgn
{
  class OutputStream
  {
    public:
      char * printed;
      void * memory;
      char * ptr;
      const char * fmt;
      int plen, maxLen;

      int memLen;
      int fmtLen;
      int * cpuFlag;

      void write(const char * const s, const int start, const int end);
    public:
      void init(const int maxSize = 8192);
      void destroy();

      __device__ __host__ OutputStream & format(const char * const s);
      __device__ __host__ OutputStream & arg(const          char          p);
      __device__ __host__ OutputStream & arg(const unsigned char          p);
      __device__ __host__ OutputStream & arg(const          short         p);
      __device__ __host__ OutputStream & arg(const unsigned short         p);
      __device__ __host__ OutputStream & arg(const          int           p);
      __device__ __host__ OutputStream & arg(const unsigned int           p);
      __device__ __host__ OutputStream & arg(const          long long     p);
      __device__ __host__ OutputStream & arg(const unsigned long long     p);
      __device__ __host__ OutputStream & arg(const float                  p);
      __device__ __host__ OutputStream & arg(const double                 p);
      __device__ __host__ OutputStream & arg(const          char * const  p);
      __device__ __host__ OutputStream & arg(const unsigned char * const  p);
      __device__ __host__ OutputStream & arg(const          void * const  p);
      __device__ __host__ OutputStream & args(const void * const arguments, const int len);
      __device__ __host__ OutputStream & finish();

      __device__ __host__ OutputStream & flush();
  };
}

#endif
