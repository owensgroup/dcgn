#include <dcgn/BroadcastRequest.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>

#include <cstring>
#include <cstdio>

namespace dcgn
{
  bool BroadcastRequest::canBeMaster() const
  {
    // we can only be the master if this node is not the root of the broadcast (first clause, in
    // which case any of the threads on the machine can be the machine's proxy) or, if this node
    // is the root, this specific thread is the root (second clause).
    return mpiWorker->getMPIRankByTarget(root) != mpiWorker->getMPIRank() || root == id;
  }
  void BroadcastRequest::performCollectiveGlobal()
  {
    profiler->add("performing global broadcast of %d %cB.", numBytes >= 1048576 ? numBytes / 1048576 : numBytes / 1024, numBytes >= 1048576 ? 'M' : 'k');
    double t = mpiWorker->wallTime();
    MPI_Bcast(buf, numBytes, MPI_BYTE, mpiWorker->getMPIRankByTarget(root), MPI_COMM_WORLD);
    // memset(buf, 0, numBytes);
    t = mpiWorker->wallTime() - t;
    profiler->add("done performing global broadcast of %d %cB, took %f seconds.", numBytes >= 1048576 ? numBytes / 1048576 : numBytes / 1024, numBytes >= 1048576 ? 'M' : 'k', t);
  }
  void BroadcastRequest::performCollectiveLocal()
  {
    profiler->add("performing local broadcast of %d %cB to %d buffers.", numBytes >= 1048576 ? numBytes / 1048576 : numBytes / 1024, numBytes >= 1048576 ? 'M' : 'k', (int)localReqs.size());
    double t = mpiWorker->wallTime();
    for (int i = 0; i < static_cast<int>(localReqs.size()); ++i)
    {
      BroadcastRequest * br = dynamic_cast<BroadcastRequest * >(localReqs[i]);
      memcpy(br->buf, buf, numBytes);
    }
    t = mpiWorker->wallTime() - t;
    profiler->add("done performing local broadcast of %d %cB, took %f seconds to %d buffers.", numBytes >= 1048576 ? numBytes / 1048576 : numBytes / 1024, numBytes >= 1048576 ? 'M' : 'k', (int)localReqs.size(), t);
  }
  BroadcastRequest::BroadcastRequest(const Target globalID, const Target pRoot, void * const buffer, const int pNumBytes, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar)
    : CollectiveRequest(REQUEST_TYPE_BROADCAST, globalID, pFinished, pError, pMutex, pCondVar), root(pRoot), buf(buffer), numBytes(pNumBytes)
  {
  }
  BroadcastRequest::~BroadcastRequest()
  {
    profiler->add("Completed broadcast (rank=%lld).", id);
  }

  void BroadcastRequest::profileStart()
  {
    profiler->add("Servicing broadcast (rank=%lld) of %d bytes (buf=%p) from root %lld.", id, numBytes, buf, root);
  }
  std::string BroadcastRequest::toString() const
  {
    char buf[1024];
    sprintf(buf, "BroadcastRequest(id=%lld, |localReqs|=%d, root=%lld, count=%d)", getID(), static_cast<int>(localReqs.size()), root, numBytes);
    return buf;
  }
}
