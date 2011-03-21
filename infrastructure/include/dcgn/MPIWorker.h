#ifndef __DCGN_MPIWORKER_H__
#define __DCGN_MPIWORKER_H__

#include <dcgn/dcgn.h>

#include <mpi.h>

#include <dcgn/ConditionVariable.h>
#include <dcgn/IORequest.h>
#include <dcgn/Mutex.h>
#include <dcgn/Request.h>
#include <dcgn/Thread.h>

#include <list>
#include <vector>

namespace dcgn
{
  class MPIWorker // : public IOWorker
  {
    protected:
      Thread thread;
      Mutex qMutex;
      ConditionVariable qCondVar;
      std::list<IORequest * > pendingReqs;
      std::vector<IORequest * > inFlightReqs;
      std::vector<int> mpiRanks, cpuRanks, gpuRanks;
      int rankStart;
      int timeout;
      double lastEvent;
      bool errorHappened;
      int cpus, gpus, slots, pause, id, size;

      void killRequests(const ErrorCode errorCode);
      void loop();
      void addRequest(IORequest * const req);
      void serviceRequest(IORequest * const req, bool & isShutdown);

      void createTargetMapping();
      void checkInFlightReqs();

      static void launchThread(void * param);
    public:
      MPIWorker(int * argc, char *** argv);
      // MPIWorker(const int totalLocalCPUs, const int totalLocalGPUs, const int slotsPerLocalGPU, const int pollPauseMS, int * argc, char *** argv);
      ~MPIWorker();

      void setCPUInfo(const int totalLocalCPUs);
      void setGPUInfo(const int totalLocalGPUs, const int slotsPerGPU);
      void setPauseTime(const int ms);
      void start();
      int getPauseTime() const;
      int getMPIRank() const;
      int getMPISize() const;
      int getMPIRankByTarget(const Target target) const;
      int getLocalSize() const;
      int getGlobalSize() const;
      int getTargetForCPU(const int cpuIndex) const;
      int getTargetForGPU(const int gpuIndex, const int slot) const;
      int getTargetForLocalCPU(const int cpuIndex) const;
      int getTargetForLocalGPU(const int gpuIndex, const int slot) const;

      const std::vector<int> & getMPIRanks() const;
      const std::vector<int> & getCPURanks() const;
      const std::vector<int> & getGPURanks() const;

      double wallTime() const;

      void send           (const void * const buffer, const int numBytes, const Target src, const Target dst, CommStatus * const stat, AsyncRequest * const req, const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar);
      void recv           (      void * const buffer, const int numBytes, const Target src, const Target dst, CommStatus * const stat, AsyncRequest * const req, const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar);
      void sendRecvReplace(      void * const buffer, const int numBytes, const Target src, const Target dst, CommStatus * const stat, AsyncRequest * const req, const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar);
      void barrier(const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar);
      void broadcast(const Target root, void * const buf, const int numBytes, const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar);

      void shutdown();
      void abort(const ErrorCode errorCode);

      bool isIdle();
      void waitForShutdown();

      int getGlobalCPUCount() const;
      int getGlobalGPUCount() const;
  };
  extern MPIWorker * mpiWorker;
}

#endif
