#ifndef __DCGN_RECVREQUEST_H__
#define __DCGN_RECVREQUEST_H__

#include <dcgn/IORequest.h>
#include <dcgn/dcgn.h>

#include <mpi.h>

namespace dcgn
{
  class RecvRequest : public IORequest
  {
    protected:
      Target id, src;
      void * buf;
      int numBytes;
      bool local, init;
      CommStatus * commStat;
      AsyncRequest * commReq;
      MPI_Status stat;
      MPI_Request req;

      bool checkLocalRequests(std::vector<IORequest * > & ioRequests);
      void probeMPI();
      bool pollLocal(std::vector<IORequest * > & ioRequests);
    public:
      RecvRequest(const Target globalID,
                  const Target pSrc, void * const pBuf, const int pNumBytes, CommStatus * const pStat, AsyncRequest * const pReq,
                  bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar);
      virtual ~RecvRequest();

      inline Target getID() const { return id; }
      inline Target getSource() const { return src; }
      inline void * getBuffer() const { return buf; }
      inline int getMaxBytes() const { return numBytes; }

      void setFinished(const int byteCount, const Target source);

      virtual void profileStart();

      virtual bool poll(std::vector<IORequest * > & ioRequests);
      virtual std::string toString() const;
  };
}

#endif
