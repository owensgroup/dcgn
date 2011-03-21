#include <dcgn/SendRecvRequest.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/RecvRequest.h>
#include <dcgn/SendRequest.h>

#include <cstring>
#include <cstdio>

namespace dcgn
{
  bool SendRecvRequest::recvCheckLocalRequests(std::vector<IORequest * > & ioRequests)
  {
    for (int i = 0; i < static_cast<int>(ioRequests.size()); ++i)
    {
      SendRecvRequest * srReq = dynamic_cast<SendRecvRequest * >(ioRequests[i]);
      SendRequest * sendReq = dynamic_cast<SendRequest * >(ioRequests[i]);
      if (sendReq && sendReq->getDestination() == id && (src == ANY_SOURCE || src == sendReq->getID()))
      {
        src = sendReq->getID();
        initRecv = localRecv = true;
        return recvPoll(ioRequests);
      }
      if (srReq && srReq->getDestination() == id && !srReq->finishedSend() && (src == ANY_SOURCE || src == srReq->getID()))
      {
        src = srReq->getID();
        initRecv = localRecv = true;
        return recvPoll(ioRequests);
      }
    }
    return false;
  }
  void SendRecvRequest::recvProbeMPI()
  {
    int ok;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &ok, &recvStat);
    if (ok) // So there *IS* a message out there.
    {
      if ((recvStat.MPI_TAG & 0xFFFF) != id) // The *next* message is not meant for us, so now we have to
      {                                      // search through all the pending messages.
        ok = 0;
        for (Target t = 0; t < mpiWorker->getGlobalSize(); ++t)
        {
          MPI_Iprobe(mpiWorker->getMPIRankByTarget(t), (t << 16) | id, MPI_COMM_WORLD, &ok, &recvStat);
          if (ok)
          {
            src = (recvStat.MPI_TAG & 0xFFFF0000) >> 16;
            profiler->add("Changing source of recv from ANY_SRC to %lld.\n", src);
            break;
          }
        }
      }
      else
      {
        src = (recvStat.MPI_TAG & 0xFFFF0000) >> 16;
      }
      if (ok)
      {
        MPI_Irecv(recvBuf, recvdBytes, MPI_BYTE, recvStat.MPI_SOURCE, recvStat.MPI_TAG, MPI_COMM_WORLD, &recvReq);
        initRecv = true;
      }
    }
  }
  bool SendRecvRequest::recvPollLocal(std::vector<IORequest * > & ioRequests)
  {
    for (int i = 0; i < static_cast<int>(ioRequests.size()); )
    {
      SendRecvRequest * srReq = dynamic_cast<SendRecvRequest * >(ioRequests[i]);
      SendRequest * sendReq = dynamic_cast<SendRequest * >(ioRequests[i]);
      if (sendReq && sendReq->getDestination() == id && (src == ANY_SOURCE || src == sendReq->getID()))
      {
        if (sendReq->getByteCount() > recvdBytes)
        {
          fprintf(stderr, "Mismatched send/recv. Trying to send more bytes than available for reception, aborting.\n");
          fflush(stderr);
          MPI_Abort(MPI_COMM_WORLD, 0);
        }
        src = sendReq->getID();
        recvdBytes = sendReq->getByteCount();
        memcpy(recvBuf, sendReq->getBuffer(), recvdBytes);
        delete sendReq;
        ioRequests.erase(ioRequests.begin() + i);
        return true;
      }
      else if (srReq && srReq->getDestination() == id && !srReq->finishedSend() && (src == ANY_SOURCE || src == srReq->getID()))
      {
        if (srReq->getSendByteCount() > numBytes)
        {
          fprintf(stderr, "Mismatched send/recv. Trying to send more bytes than available for reception, aborting.\n");
          fflush(stderr);
          MPI_Abort(MPI_COMM_WORLD, 0);
        }
        src = srReq->getID();
        recvdBytes = srReq->getSendByteCount();
        memcpy(recvBuf, srReq->getSendBuffer(), recvdBytes);
        srReq->setFinishedSend();
        if (srReq->finishedRecv())
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
  bool SendRecvRequest::recvPoll(std::vector<IORequest * > & ioRequests)
  {
    if (!initRecv && !localRecv)
    {
      if (src == ANY_SOURCE)
      {
        if (recvCheckLocalRequests(ioRequests)) return true;
        if (!localRecv) recvProbeMPI();
        if (!initRecv && !localRecv) return false;
      }
      else
      {
        MPI_Irecv(recvBuf, recvdBytes, MPI_BYTE, mpiWorker->getMPIRankByTarget(dst), static_cast<int>((src << 16) | id), MPI_COMM_WORLD, &recvReq);
        initRecv = true;
      }
    }
    if (localRecv)
    {
      return doneRecv = recvPollLocal(ioRequests);
    }
    int ok;
    MPI_Test(&recvReq, &ok, &recvStat);
    if (ok) MPI_Get_count(&recvStat, MPI_BYTE, &recvdBytes);
    return doneRecv = (ok != 0);
  }
  bool SendRecvRequest::sendPollLocal(std::vector<IORequest * > & ioRequests)
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
        memcpy(recvReq->getBuffer(), sendBuf, numBytes);
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
        memcpy(srReq->getRecvBuffer(), sendBuf, numBytes);
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
  bool SendRecvRequest::sendPoll(std::vector<IORequest * > & ioRequests)
  {
    if (!initSend && !localSend)
    {
      MPI_Isend(const_cast<void * >(sendBuf), numBytes, MPI_BYTE, mpiWorker->getMPIRankByTarget(dst), static_cast<int>((id << 16) | dst), MPI_COMM_WORLD, &sendReq);
      initSend = true;
    }
    if (localSend)
    {
      return doneSend = sendPollLocal(ioRequests);
    }
    int ok;
    MPI_Test(&sendReq, &ok, &sendStat);
    return doneSend = (ok != 0);
  }

  SendRecvRequest::SendRecvRequest(const bool deleteSendBuf, const Target globalID,
                                   const Target pDst, const void * const pSendBuf, const int pSendBytes,
                                   const Target pSrc,       void * const pRecvBuf, const int pRecvBytes, CommStatus * const pCommStat, AsyncRequest * const pCommReq,
                                   bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar)
    : IORequest(deleteSendBuf ? REQUEST_TYPE_SEND_RECV_REPLACE : REQUEST_TYPE_SEND_RECV, pFinished, pError, pMutex, pCondVar),
      id(globalID),
      src(pSrc), dst(pDst), sendBuf(pSendBuf), recvBuf(pRecvBuf), numBytes(pSendBytes), recvdBytes(pRecvBytes),
      delSend(deleteSendBuf), localSend(false), localRecv(false), initSend(false), initRecv(false), doneSend(false), doneRecv(false),
      commStat(pCommStat), commReq(pCommReq)
  {
  }
  SendRecvRequest::~SendRecvRequest()
  {
    profiler->add("Completed sendRecvReplace (rank=%lld) of %d bytes (buf=%p) from %lld, to %lld.", id, numBytes, sendBuf, src, dst);
    if (commStat)
    {
      commStat->src = src;
      commStat->dst = id;
      commStat->numBytes = recvdBytes;
      commStat->errorCode = DCGN_ERROR_NONE;
    }
    if (commReq)
    {
      commReq->completed = true;
      // we assume that the commStat varible will be &commReq->stat here.
    }
    if (delSend && sendBuf) delete [] reinterpret_cast<const char * >(sendBuf);
  }

  void SendRecvRequest::profileStart()
  {
    if (delSend)  profiler->add("Servicing sendRecvReplace  (rank=%lld) with %d bytes (buf=%p) from=%lld, to %lld.", id, numBytes, recvBuf, src, dst);
    else          profiler->add("Servicing sendRecv         (rank=%lld) with %d bytes (buf=%p) from=%lld, to %lld.", id, numBytes, recvBuf, src, dst);
  }
  bool SendRecvRequest::poll(std::vector<IORequest * > & ioRequests)
  {
    if (!doneSend) sendPoll(ioRequests);
    if (!doneRecv) recvPoll(ioRequests);
    return doneSend && doneRecv;
  }
  std::string SendRecvRequest::toString() const
  {
    char buf[1024];
    sprintf(buf, "SendRecvRequest(rank=%lld, dst=%lld, sendCount=%d, src=%lld, recvCount=%d)", id, dst, numBytes, src, recvdBytes);
    return buf;
  }
}
