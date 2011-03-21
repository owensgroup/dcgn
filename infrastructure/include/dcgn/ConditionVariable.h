#ifndef __DCGN_CONDITIONVARIABLE_H__
#define __DCGN_CONDITIONVARIABLE_H__

#include <pthread.h>

namespace dcgn
{
  class Mutex;
  class ConditionVariable
  {
    protected:
      pthread_cond_t condVar;
      int waitingCount;
    public:
      ConditionVariable();
      ~ConditionVariable();

      int getWaitingCount();
      void wait(Mutex & m);
      void signal();
      void broadcast();

      pthread_cond_t & getNativeResource();
  };
}

#endif
