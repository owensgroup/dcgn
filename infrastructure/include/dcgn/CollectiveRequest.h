#ifndef __DCGN_COLLECTIVEREQUEST_H__
#define __DCGN_COLLECTIVEREQUEST_H__

#include <dcgn/IORequest.h>
#include <dcgn/dcgn.h>
#include <vector>

namespace dcgn
{
  class CollectiveRequest : public IORequest
  {
    protected:
      Target id;
      std::vector<IORequest * > localReqs;

      bool consolidate(std::vector<IORequest * > & ioRequests);

      virtual bool canBeMaster() const = 0;
      virtual void performCollectiveGlobal() = 0;
      virtual void performCollectiveLocal() = 0;
    public:
      CollectiveRequest(const int reqType, const Target globalID, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar);
      virtual ~CollectiveRequest();

      virtual void profileStart() = 0;

      inline Target getID() const { return id; }

      virtual bool poll(std::vector<IORequest * > & ioRequests);
  };
}

#endif
