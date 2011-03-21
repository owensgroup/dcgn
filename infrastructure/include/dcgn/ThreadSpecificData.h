#ifndef __DCGN_THREADSPECIFICDATA_H__
#define __DCGN_THREADSPECIFICDATA_H__

#include <pthread.h>

namespace dcgn
{
  class ThreadSpecificData
  {
    protected:
      pthread_key_t key;
    public:
      ThreadSpecificData();
      ~ThreadSpecificData();

      void * getValue();
      void setValue(void * const p);

      pthread_key_t & getNativeResource();
  };
}

#endif
