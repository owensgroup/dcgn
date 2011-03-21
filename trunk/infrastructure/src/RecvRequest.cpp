#include <dcgn/RecvRequest.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/SendRequest.h>
#include <dcgn/SendRecvRequest.h>

#include <typeinfo>
#include <cstdio>
#include <cstring>

namespace dcgn
{
  bool RecvRequest::checkLocalRequests(std::vector<IORequest * > & ioRequests)
  {
    for (int i = 0; i < static_cast<int>(ioRequests.size()); ++i)
    {
      SendRecvRequest * srReq = dynamic_cast<SendRecvRequest * >(ioRequests[i]);
      SendRequest * sendReq = dynamic_cast<SendRequest * >(ioRequests[i]);
      if (sendReq && sendReq->getDestination() == id && (src == ANY_SOURCE || src == sendReq->getID()))
      {
        src = sendReq->getID();
        init = local = true;
        return poll(ioRequests);
      }
      if (srReq && srReq->getDestination() == id && !srReq->finishedSend() && (src == ANY_SOURCE || src == srReq->getID()))
      {
        src = srReq->getID();
        init = local = true;
        return poll(ioRequests);
      }
    }
    return false;
  }
  void RecvRequest::probeMPI()
  {
    int ok;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &ok, &stat);
    if (ok) // So there *IS* a message out there.
    {
      if ((stat.MPI_TAG & 0xFFFF) != id) // The *next* message is not meant for us, so now we have to
      {                                  // search through all the pending messages.
        ok = 0;
        for (Target t = 0; t < mpiWorker->getGlobalSize(); ++t)
        {
          MPI_Iprobe(mpiWorker->getMPIRankByTarget(t), (t << 16) | id, MPI_COMM_WORLD, &ok, &stat);
          if (ok)
          {
            src = (stat.MPI_TAG & 0xFFFF0000) >> 16;
            profiler->add("Changing source of recv from ANY_SRC to %lld.\n", src);
            break;
          }
        }
      }
      else
      {
        src = (stat.MPI_TAG & 0xFFFF0000) >> 16;
      }
      if (ok)
      {
        MPI_Irecv(buf, numBytes, MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &req);
        init = true;
      }
    }
  }
  bool RecvRequest::pollLocal(std::vector<IORequest * > & ioRequests)
  {
    for (int i = 0; i < static_cast<int>(ioRequests.size()); )
    {
      SendRecvRequest * srReq = dynamic_cast<SendRecvRequest * >(ioRequests[i]);
      SendRequest * sendReq = dynamic_cast<SendRequest * >(ioRequests[i]);
      if (sendReq && sendReq->getDestination() == id && (src == ANY_SOURCE || src == sendReq->getID()))
      {
        if (sendReq->getByteCount() > numBytes)
        {
          fprintf(stderr, "Mismatched send/recv. Trying to send more bytes than available for reception, aborting.\n");
          fflush(stderr);
          MPI_Abort(MPI_COMM_WORLD, 0);
        }
        src = sendReq->getID();
        numBytes = sendReq->getByteCount();
        memcpy(buf, sendReq->getBuffer(), numBytes);
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
        numBytes = srReq->getSendByteCount();
        memcpy(buf, srReq->getSendBuffer(), numBytes);
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

  RecvRequest::RecvRequest(const Target globalID,
                           const Target pSrc, void * const pBuf, const int pNumBytes, CommStatus * const pStat, AsyncRequest * const pReq,
                           bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar)
    : IORequest(REQUEST_TYPE_RECV, pFinished, pError, pMutex, pCondVar), id(globalID), src(pSrc), buf(pBuf), numBytes(pNumBytes), commStat(pStat), commReq(pReq)
  {
    local = (src != ANY_SOURCE && mpiWorker->getMPIRankByTarget(src) == mpiWorker->getMPIRank());
    init = false;
  }
  RecvRequest::~RecvRequest()
  {
    profiler->add("Completed recv (rank=%lld) of %d bytes (buf=%p) from %lld.", id, numBytes, buf, src);
    if (commStat)
    {
      commStat->src = src;
      commStat->dst = id;
      commStat->numBytes = numBytes;
      commStat->errorCode = DCGN_ERROR_NONE;
    }
    if (commReq)
    {
      commReq->completed = true;
      // we assume that the commStat varible will be &commReq->stat here.
    }
  }

  void RecvRequest::setFinished(const int byteCount, const Target source)
  {
    numBytes = byteCount;
    src = source;
  }

  void RecvRequest::profileStart()
  {
    if (src == ANY_SOURCE)  profiler->add("Servicing recv (rank=%lld) with %d bytes (buf=%p) from ANY_SRC.",  id, numBytes, buf);
    else                    profiler->add("Servicing recv (rank=%lld) with %d bytes (buf=%p) from %lld.",     id, numBytes, buf, src);
  }
  bool RecvRequest::poll(std::vector<IORequest * > & ioRequests)
  {
    if (!init && !local)
    {
      if (src == ANY_SOURCE)
      {
        if (checkLocalRequests(ioRequests)) return true;
        if (!local) probeMPI();
        if (!init && !local) return false;
      }
      else
      {
        MPI_Irecv(buf, numBytes, MPI_BYTE, mpiWorker->getMPIRankByTarget(src), static_cast<int>((src << 16) | id), MPI_COMM_WORLD, &req);
        init = true;
      }
    }
    if (local)
    {
      if (!init)
      {
        init = true;
      }
      return pollLocal(ioRequests);
    }
    int ok;
    MPI_Test(&req, &ok, &stat);
    if (ok) MPI_Get_count(&stat, MPI_BYTE, &numBytes);
    return ok != 0;
  }
  std::string RecvRequest::toString() const
  {
    char buf[1024];
    sprintf(buf, "RecvRequest(rank=%lld, src=%lld, count=%d)", id, src, numBytes);
    return buf;
  }
}
