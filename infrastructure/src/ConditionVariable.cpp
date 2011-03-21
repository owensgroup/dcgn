#include <dcgn/ConditionVariable.h>
#include <dcgn/Mutex.h>

namespace dcgn
{
  ConditionVariable::ConditionVariable()
  {
    pthread_cond_init(&condVar, 0);
  }
  ConditionVariable::~ConditionVariable()
  {
    pthread_cond_destroy(&condVar);
  }

  int ConditionVariable::getWaitingCount()
  {
    return waitingCount;
  }
  void ConditionVariable::wait(Mutex & m)
  {
    ++waitingCount;
    pthread_cond_wait(&condVar, &m.getNativeResource());
  }
  void ConditionVariable::signal()
  {
    --waitingCount;
    pthread_cond_signal(&condVar);
  }
  void ConditionVariable::broadcast()
  {
    waitingCount = 0;
    pthread_cond_broadcast(&condVar);
  }

  pthread_cond_t & ConditionVariable::getNativeResource()
  {
    return condVar;
  }
}
