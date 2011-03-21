#include <dcgn/IORequest.h>
#include <dcgn/ConditionVariable.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Mutex.h>
#include <dcgn/Profiler.h>

namespace dcgn
{
  IORequest::IORequest(const int reqType, bool * const pFinished, ErrorCode * const pError, Mutex * const pMutex, ConditionVariable * const pCondVar)
    : type(reqType), finished(pFinished), error(pError), mutex(pMutex), condVar(pCondVar)
  {
    if (error) *error = DCGN_ERROR_NONE;
    //
    startTime = mpiWorker->wallTime();
    firstPollTime = commStartTime = commEndTime = localStartTime = localEndTime = signalTime = destTime = -1.0;
  }
  IORequest::~IORequest()
  {
    signalTime = mpiWorker->wallTime();
    if (finished) *finished = true;
    if (mutex && condVar)
    {
      mutex->lock();
      condVar->signal();
      mutex->unlock();
    }
    if (commStartTime >= 0.0)
    {
      // profiler->addTime(type, startTime, firstPollTime, commStartTime, commEndTime, localStartTime, localEndTime, signalTime, destTime);
      destTime = mpiWorker->wallTime();
#if 0
      fprintf(stderr, "fullTime: %14.10f\n"
                      "  time to first poll:        %14.10f\n"
                      "  time from poll to comm:    %14.10f\n"
                      "  global comm time:          %14.10f\n"
                      "  local comm time:           %14.10f\n"
                      "  time from comm to signal:  %14.10f\n"
                      "  time from signal to dtor:  %14.10f\n",
                      destTime - startTime,
                      firstPollTime - startTime,
                      commStartTime - firstPollTime,
                      commEndTime - commStartTime,
                      localEndTime - localStartTime,
                      signalTime - localEndTime,
                      destTime - signalTime);
      fflush(stderr);
#endif
    }
  }
}
