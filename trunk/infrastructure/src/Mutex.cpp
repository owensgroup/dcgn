#include <dcgn/Mutex.h>

namespace dcgn
{
  Mutex::Mutex() : locked(false)
  {
    pthread_mutex_init(&mutex, NULL);
  }
  Mutex::~Mutex()
  {
    pthread_mutex_destroy(&mutex);
  }

  bool Mutex::isLocked() const
  {
    return locked;
  }
  void Mutex::lock()
  {
    pthread_mutex_lock(&mutex);
    locked = true;
  }
  void Mutex::unlock()
  {
    pthread_mutex_unlock(&mutex);
    locked = false;
  }

  void Mutex::conditionVariableWait()
  {
    locked = false;
  }
  pthread_mutex_t & Mutex::getNativeResource()
  {
    return mutex;
  }
}
