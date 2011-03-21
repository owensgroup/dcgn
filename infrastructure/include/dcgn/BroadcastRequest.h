#ifndef __DCGN_BROADCASTREQUEST_H__
#define __DCGN_BROADCASTREQUEST_H__

#include <dcgn/CollectiveRequest.h>
#include <vector>

namespace dcgn
{
  class BroadcastRequest : public CollectiveRequest
  {
    protected:
      Target root;
      void * const buf;
      int numBytes;

      virtual bool canBeMaster() const;
      virtual void performCollectiveGlobal();
      virtual void performCollectiveLocal();
    public:
      BroadcastRequest(const Target globalID, const Target pRoot, void * const buffer, const int pNumBytes, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar);
      virtual ~BroadcastRequest();

      virtual void profileStart();

      inline Target getRoot()       const { return root;      }
      inline void * getBuffer()     const { return buf;       }
      inline int    getByteCount()  const { return numBytes;  }

      virtual std::string toString() const;
  };
}

#endif
