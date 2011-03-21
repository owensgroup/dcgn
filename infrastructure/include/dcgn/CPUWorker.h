#ifndef __DCGN_CPUWORKER_H__
#define __DCGN_CPUWORKER_H__

#include <dcgn/dcgn.h>

#include <dcgn/ConditionVariable.h>
#include <dcgn/Mutex.h>
#include <dcgn/Request.h>
#include <dcgn/Thread.h>
#include <dcgn/ThreadSpecificData.h>

#include <list>
#include <vector>

namespace dcgn
{
  class CPUWorker
  {
    protected:
      static ThreadSpecificData workerHandle;
      int cpuID, localID, globalID;
      bool servingRequest, errorHappened;
      OutputStream outputStream;
      Thread thread;
      Mutex qMutex, ioMutex;
      ConditionVariable qCondVar, ioCondVar;
      std::list<Request * > pendingReqs;

      void checkError(const ErrorCode code);
      void loop();
      void addRequest(Request * const req);
      void serviceRequest(Request * const req, bool & isShutdown);

      static void launchThread(void * param);
    public:
      CPUWorker(const int cpuThreadIndex);
      ~CPUWorker();

      void setGlobalID(const int id);
      void start();

      void scheduleKernel(const CPUKernelFunction kernelFunc, void * const param);
      void shutdown();

      void abort(const ErrorCode errorCode);

      bool isIdle();
      void waitForShutdown();

      static void send   (const Target dst, const void * const buffer, const int numBytes);
      static void recv   (const Target src,       void * const buffer, const int maxBytes, CommStatus * const stat);
      static void sendRecvReplace(const Target dst, const Target src, void * const buffer, const int numBytes, CommStatus * const stat);
      static void barrier();
      static void broadcast(const Target root, void * const bytes, const int numBytes);
      static void asyncSend(const Target dst, const void * const buffer, const int numBytes, AsyncRequest * const req);
      static void asyncRecv(const Target src,       void * const buffer, const int maxBytes, AsyncRequest * const req);
      static Target getRank();
      static bool isCPUWorkerThread();
      static OutputStream & output();
  };
  extern std::vector<CPUWorker * > cpuWorkers;
}

#endif
