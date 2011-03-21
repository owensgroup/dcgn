#include <dcgn/BarrierRequest.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <cstdio>
#include <vector>

namespace dcgn
{
  bool BarrierRequest::canBeMaster() const
  {
    return true;
  }
  void BarrierRequest::performCollectiveGlobal()
  {
    MPI_Barrier(MPI_COMM_WORLD);
  }
  void BarrierRequest::performCollectiveLocal()
  {
  }
  BarrierRequest::BarrierRequest(const Target globalID, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar)
    : CollectiveRequest(REQUEST_TYPE_BARRIER, globalID, pFinished, pError, pMutex, pCondVar)
  {
  }
  BarrierRequest::~BarrierRequest()
  {
    profiler->add("Completed barrier (rank=%lld).", id);
  }

  void BarrierRequest::profileStart()
  {
    profiler->add("Servicing barrier (rank= %lld).", id);
  }
  std::string BarrierRequest::toString() const
  {
    char buf[1024];
    sprintf(buf, "BarrierRequest(|localReqs|=%d)", static_cast<int>(localReqs.size()));
    return buf;
  }
}
