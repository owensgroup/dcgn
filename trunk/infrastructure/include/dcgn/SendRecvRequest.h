#ifndef __DCGN_SENDRECVREQUEST_H__
#define __DCGN_SENDRECVREQUEST_H__

#include <dcgn/dcgn.h>
#include <dcgn/IORequest.h>

#include <mpi.h>

namespace dcgn
{
  class SendRecvRequest : public IORequest
  {
    protected:
      Target id, src, dst;
      const void * sendBuf;
      void * recvBuf;
      int numBytes;
      int recvdBytes;
      bool delSend, localSend, localRecv, initSend, initRecv, doneSend, doneRecv;
      CommStatus * commStat;
      AsyncRequest * commReq;
      MPI_Status sendStat, recvStat;
      MPI_Request sendReq, recvReq;


      bool recvCheckLocalRequests(std::vector<IORequest * > & ioRequests);
      void recvProbeMPI();
      bool recvPollLocal(std::vector<IORequest * > & ioRequests);
      bool recvPoll(std::vector<IORequest * > & ioRequests);

      bool sendPollLocal(std::vector<IORequest * > & ioRequests);
      bool sendPoll(std::vector<IORequest * > & ioRequests);
    public:
      SendRecvRequest(const bool deleteSendBuf, const Target globalID,
                      const Target pDst, const void * const pSendBuf, const int pSendBytes,
                      const Target pSrc,       void * const pRecvBuf, const int pRecvBytes, CommStatus * const pCommStat, AsyncRequest * const pCommReq,
                      bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar);
      virtual ~SendRecvRequest();

      inline Target       getID()             const { return id; }
      inline Target       getSource()         const { return id; }
      inline const void * getSendBuffer()     const { return sendBuf; }
      inline int          getSendByteCount()  const { return numBytes; }
      inline bool         finishedSend()      const { return doneSend; }
      inline Target       getDestination()    const { return dst; }
      inline       void * getRecvBuffer()     const { return recvBuf; }
      inline int          getRecvByteCount()  const { return recvdBytes; }
      inline bool         finishedRecv()      const { return doneRecv; }

      inline void setFinishedSend()
      {
        doneSend = true;
      }
      inline void setFinishedRecv(const int byteCount, const Target pSrc)
      {
        recvdBytes = byteCount;
        src = pSrc;
        doneRecv = true;
      }

      virtual void profileStart();
      virtual bool poll(std::vector<IORequest * > & ioRequests);
      virtual std::string toString() const;
  };
}

#endif
