#include <dcgn/SendRequest.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/RecvRequest.h>
#include <dcgn/SendRecvRequest.h>

#include <cstring>
#include <cstdio>

namespace dcgn
{

  bool SendRequest::pollLocal(std::vector<IORequest * > & ioRequests)
  {
    for (int i = 0; i < static_cast<int>(ioRequests.size()); )
    {
      SendRecvRequest * srReq = dynamic_cast<SendRecvRequest * >(ioRequests[i]);
      RecvRequest * recvReq = dynamic_cast<RecvRequest * >(ioRequests[i]);
      if (recvReq && dst == recvReq->getID() && (recvReq->getSource() == ANY_SOURCE || recvReq->getSource() == id))
      {
        if (numBytes > recvReq->getMaxBytes())
        {
          fprintf(stderr, "Mismatched send/recv. Trying to send more bytes than available for reception, aborting.\n");
          fflush(stderr);
          MPI_Abort(MPI_COMM_WORLD, 0);
        }
        memcpy(recvReq->getBuffer(), buf, numBytes);
        recvReq->setFinished(numBytes, id);
        delete recvReq;
        ioRequests.erase(ioRequests.begin() + i);
        return true;
      }
      else if (srReq && dst == srReq->getID() && (srReq->getSource() == ANY_SOURCE || srReq->getSource() == id) && !srReq->finishedRecv())
      {
        if (numBytes > srReq->getRecvByteCount())
        {
          fprintf(stderr, "Mismatched send/recv. Trying to send more bytes than available for reception, aborting.\n");
          fflush(stderr);
          MPI_Abort(MPI_COMM_WORLD, 0);
        }
        memcpy(srReq->getRecvBuffer(), buf, numBytes);
        srReq->setFinishedRecv(numBytes, id);
        if (srReq->finishedSend())
        {
          delete srReq;
          ioRequests.erase(ioRequests.begin() + i);
        }
        return true;
      }
      ++i;
    }
    return false;
  }
  SendRequest::SendRequest(const Target globalID, const Target pDst, const void * const pBuf, const int pNumBytes, CommStatus * pCommStat, AsyncRequest * const pCommReq, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar)
    : IORequest(REQUEST_TYPE_SEND, pFinished, pError, pMutex, pCondVar), id(globalID), dst(pDst), buf(pBuf), numBytes(pNumBytes), commStat(pCommStat), commReq(pCommReq)
  {
    local = (mpiWorker->getMPIRankByTarget(dst) == mpiWorker->getMPIRank());
    init = false;
  }
  SendRequest::~SendRequest()
  {
    profiler->add("Completed send (rank=%lld) of %d bytes (buf=%p) to %lld.", id, numBytes, buf, dst);
    if (commStat)
    {
      commStat->src = id;
      commStat->dst = dst;
      commStat->numBytes = numBytes;
      commStat->errorCode = DCGN_ERROR_NONE;
    }
    if (commReq)
    {
      commReq->completed = true;
      // we assume that the commStat varible will be &commReq->stat here.
    }
  }

  void SendRequest::profileStart()
  {
    profiler->add("Servicing send (rank=%lld) with %d bytes (buf=%p) to %lld.", id, numBytes, buf, dst);
  }
  bool SendRequest::poll(std::vector<IORequest * > & ioRequests)
  {
    if (!init && !local)
    {
      MPI_Isend(const_cast<void * >(buf), numBytes, MPI_BYTE, mpiWorker->getMPIRankByTarget(dst), static_cast<int>((id << 16) | dst), MPI_COMM_WORLD, &req);
      init = true;
    }
    if (local)
    {
      init = true;
      return pollLocal(ioRequests);
    }
    int ok;
    MPI_Test(&req, &ok, &stat);
    return ok != 0;
  }
  std::string SendRequest::toString() const
  {
    char buf[1024];
    sprintf(buf, "SendRequest(rank=%lld, dst=%lld, count=%d)", id, dst, numBytes);
    return buf;
  }
}
