#include <dcgn/ThreadSpecificData.h>

namespace dcgn
{
  ThreadSpecificData::ThreadSpecificData()
  {
    pthread_key_create(&key, 0);
  }
  ThreadSpecificData::~ThreadSpecificData()
  {
    pthread_key_delete(key);
  }

  void * ThreadSpecificData::getValue()
  {
    return pthread_getspecific(key);
  }
  void ThreadSpecificData::setValue(void * const p)
  {
    pthread_setspecific(key, p);
  }
  pthread_key_t & ThreadSpecificData::getNativeResource()
  {
    return key;
  }
}
