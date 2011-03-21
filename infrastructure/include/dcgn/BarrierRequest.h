#ifndef __DCGN_BARRIERREQUEST_H__
#define __DCGN_BARRIERREQUEST_H__

#include <dcgn/CollectiveRequest.h>
#include <dcgn/dcgn.h>
#include <vector>

namespace dcgn
{
  class BarrierRequest : public CollectiveRequest
  {
    protected:
      virtual bool canBeMaster() const;
      virtual void performCollectiveGlobal();
      virtual void performCollectiveLocal();
    public:
      BarrierRequest(const Target globalID, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar);
      virtual ~BarrierRequest();

      virtual void profileStart();
      virtual std::string toString() const;
  };
}

#endif
