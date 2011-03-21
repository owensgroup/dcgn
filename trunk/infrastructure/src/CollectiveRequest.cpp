#include <dcgn/CollectiveRequest.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>

#include <cstdio>

namespace dcgn
{
  bool CollectiveRequest::consolidate(std::vector<IORequest * > & ioRequests)
  {
    bool consolidated = false;
    int selfIndex = -1;
    for (int i = 0; i < static_cast<int>(ioRequests.size()); ++i)
    {
      if (ioRequests[i] == this)
      {
        selfIndex = i;
        break;
      }
    }
    for (int i = 0; i < static_cast<int>(ioRequests.size()); )
    {
      CollectiveRequest * req = dynamic_cast<CollectiveRequest * >(ioRequests[i]);
      if (ioRequests[i] == this || req == 0)
      {
        ++i;
      }
      else if (req != 0 && ioRequests[i]->getType() != getType())
      {
        fprintf(stderr, "Error, got collective requests of type %s and %s simultaneously.\n", REQUEST_STRINGS[getType()], REQUEST_STRINGS[ioRequests[i]->getType()]);
        fflush(stderr);
        dcgn::abort(DCGN_ERROR_MPI);
        return false;
      }
      else if (i < selfIndex)
      {
        std::swap(ioRequests[selfIndex], ioRequests[i]);
        return consolidate(ioRequests);
      }
      else if (!req->canBeMaster() || req->localReqs.size() <= localReqs.size())
      {
        profiler->add("%s (rank=%lld) absorbing %s (rank=%lld).", REQUEST_STRINGS[getType()], id, REQUEST_STRINGS[req->getType()], req->id);
        localReqs.push_back(ioRequests[i]);
        for (int j = 0; j < static_cast<int>(req->localReqs.size()); ++j)
        {
          profiler->add("%s (rank=%lld) absorbing %s (rank=%lld).", REQUEST_STRINGS[getType()], id, REQUEST_STRINGS[req->localReqs[j]->getType()], dynamic_cast<CollectiveRequest * >(req->localReqs[j])->id);
          localReqs.push_back(req->localReqs[j]);
        }
        req->localReqs.clear();
        ioRequests.erase(ioRequests.begin() + i);
        consolidated = true;
      }
      else
      {
        ++i;
      }
    }
    return consolidated;
  }

  CollectiveRequest::CollectiveRequest(const int reqType, const Target globalID, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar)
    : IORequest(reqType, pFinished, pError, pMutex, pCondVar), id(globalID)
  {
  }
  CollectiveRequest::~CollectiveRequest()
  {
    for (int i = 0; i < static_cast<int>(localReqs.size()); ++i) delete localReqs[i];
  }
  bool CollectiveRequest::poll(std::vector<IORequest * > & ioRequests)
  {
    if (firstPollTime < 0.0) firstPollTime = mpiWorker->wallTime();
    if (static_cast<int>(localReqs.size() + 1) == mpiWorker->getLocalSize())
    {
      commStartTime = mpiWorker->wallTime();
      if (mpiWorker->getGlobalSize() > mpiWorker->getLocalSize())
      {
        performCollectiveGlobal();
      }
      commEndTime = localStartTime = mpiWorker->wallTime();
      performCollectiveLocal();
      localEndTime = mpiWorker->wallTime();
      return true;
    }
    else if (canBeMaster())
    {
      if (consolidate(ioRequests)) return poll(ioRequests);
    }
    return false;
  }
}
