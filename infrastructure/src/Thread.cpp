#include <dcgn/Thread.h>
#include <sched.h>
#include <unistd.h>
#include <cstdio>

namespace dcgn
{
  typedef struct _ThreadData
  {
    Thread::RunFunction func;
    void * param;
    Thread * t;
  } ThreadData;

  void * Thread::run(void * p)
  {
    ThreadData * td = reinterpret_cast<ThreadData * >(p);
    td->func(td->param);
    td->t->running = false;
    delete td;
    return 0;
  }

  Thread::Thread() : running(false)
  {
  }

  Thread::~Thread()
  {
  }

  void Thread::start(RunFunction func, void * const param)
  {
    ThreadData * td = new ThreadData;
    td->func = func;
    td->param = param;
    td->t = this;
    td->t->running = true;
    pthread_create(&thread, 0, run, td);
  }

  bool Thread::isRunning() const
  {
    return running;
  }
  void Thread::waitFor()
  {
    void * dummy;
    pthread_join(thread, &dummy);
  }
  void Thread::exit()
  {
    pthread_cancel(thread);
  }

  void Thread::sleep(const int ms)
  {
    usleep(ms * 1000);
  }
  void Thread::yield()
  {
    sched_yield();
  }
}
