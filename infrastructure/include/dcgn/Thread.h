#ifndef __DCGN_THREAD_H__
#define __DCGN_THREAD_H__

#include <pthread.h>

namespace dcgn
{
  class Thread
  {
    public:
      typedef void (*RunFunction)(void * );
    protected:
      static void * run(void * p);

      pthread_t thread;
      bool running;
    public:
      Thread();
      ~Thread();

      void start(RunFunction func, void * const param);
      bool isRunning() const;
      void waitFor();
      void exit();

      static void sleep(const int ms);
      static void yield();
  };
}

#endif
