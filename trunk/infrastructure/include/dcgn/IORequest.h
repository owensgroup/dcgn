#ifndef __DCGN_IOREQUEST_H__
#define __DCGN_IOREQUEST_H__

#include <dcgn/dcgn.h>

#include <string>
#include <vector>

namespace dcgn
{
  class Mutex;
  class ConditionVariable;
  class IORequest
  {
    protected:
      int type;
      bool * finished;
      ErrorCode * const error;
      Mutex * mutex;
      ConditionVariable * condVar;

      double startTime, firstPollTime, commStartTime, commEndTime, localStartTime, localEndTime, signalTime, destTime;
    public:
      IORequest(const int reqType, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar);
      virtual ~IORequest();

      virtual void profileStart() = 0;

      virtual bool poll(std::vector<IORequest * > & ioRequests) = 0;
      inline void setError(const ErrorCode err) { *error = err; }
      inline int getType() const { return type; }
      virtual std::string toString() const = 0;
  };
}

#endif
