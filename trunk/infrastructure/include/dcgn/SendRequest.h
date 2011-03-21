#ifndef __DCGN_SENDREQUEST_H__
#define __DCGN_SENDREQUEST_H__

#include <dcgn/dcgn.h>
#include <dcgn/IORequest.h>

#include <mpi.h>

namespace dcgn
{
  class SendRequest : public IORequest
  {
    protected:
      Target id, dst;
      const void * buf;
      int numBytes;
      bool local, init;
      CommStatus * commStat;
      AsyncRequest * commReq;
      MPI_Status stat;
      MPI_Request req;

      bool pollLocal(std::vector<IORequest * > & ioRequests);
    public:
      SendRequest(const Target globalID, const Target pDst, const void * const pBuf, const int pNumBytes, CommStatus * pCommStat, AsyncRequest * const pCommReq, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar);
      virtual ~SendRequest();

      inline Target getID() const { return id; }
      inline Target getDestination() const { return dst; }
      inline const void * getBuffer() const { return buf; }
      inline int getByteCount() const { return numBytes; }

      virtual void profileStart();
      virtual bool poll(std::vector<IORequest * > & ioRequests);
      virtual std::string toString() const;
  };
}

#endif
