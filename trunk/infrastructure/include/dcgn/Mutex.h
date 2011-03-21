#ifndef __DCGN_MUTEX_H__
#define __DCGN_MUTEX_H__

#include <pthread.h>

namespace dcgn
{
  class Mutex
  {
    protected:
      pthread_mutex_t mutex;
      bool locked;
    public:
      Mutex();
      ~Mutex();

      bool isLocked() const;
      void lock();
      void unlock();

      void conditionVariableWait();
      pthread_mutex_t & getNativeResource();
  };
}

#endif
