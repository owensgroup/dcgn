#ifndef __DCGN_GPUWORKER_H__
#define __DCGN_GPUWORKER_H__

#include <dcgn/dcgn.h>

#include <dcgn/ConditionVariable.h>
#include <dcgn/Mutex.h>
#include <dcgn/Request.h>
#include <dcgn/Thread.h>
#include <dcgn/ThreadSpecificData.h>

#include <map>
#include <list>
#include <vector>

namespace dcgn
{
  class GPUWorker
  {
    protected:
      typedef struct _InternalRequest
      {
        RequestType type;
        void * buf;
        ErrorCode error;
        GPUIORequest * localReq;
        GPUIORequest * gpuReq;
        int * inFlightFlag;
        Target rank;
        bool done;
      } InternalRequest;

      static ThreadSpecificData workerHandle;
      bool errorHappened;
      int timeout;
      double lastEvent;
      int slots, asyncCount, pause, deviceID, gpuID, localID, globalID;
      bool servingRequest;
      Thread thread;
      Mutex qMutex, ioMutex;
      ConditionVariable qCondVar, ioCondVar;
      std::list<Request * > pendingReqs;
      std::vector<InternalRequest * > inFlightReqs;
      size_t pageLockedMemSize;
      OutputStream outputStream, * gpuStream;
      void * pageLockedMem;
      GPUInitRequest initReq;
      GPUCleanupFunction currentDtor;
      void * currentParam;
      cudaStream_t              kernelStream, memcpyStream;
      std::map<void * , int>    usedBufs;
      std::map<void * , int>    pageLockedBufs;
      std::vector<GPUIORequest> localAsyncReqs; // local asynchronous requests.
      std::vector<GPUIORequest> localReqs;      // target to cudaMemcpy gpu requests.
      std::vector<int>          inFlight;       // used to determine if a given req is actually in flight. using ints instead of floats due to issues with bool vectors.
      std::vector<int>          inFlightAsync;  // used to determine if a given req is actually in flight. using ints instead of floats due to issues with bool vectors.

      void checkError(const ErrorCode code);
      void loop();
      void addRequest(Request * const req);
      void serviceRequest(Request * const req, bool & isShutdown);

      void ensurePageLockedMemSize(const size_t size);
      void copyFromDevice(const bool pageLocked, void * const cpuMem, const void * const gpuMem, const size_t size);
      void copyToDevice  (const bool pageLocked, void * const gpuMem, const void * const cpuMem, const size_t size);

      void * findPageLockedBuffer(const int size);
      void releasePageLockedBuffer(void * const buffer);
      void copyUpAndInit(InternalRequest * const req, GPUIORequest * const localReq);

      void pollRequestsSub(std::vector<GPUIORequest> & reqs, GPUIORequest * gpuReqs, std::vector<int> & inFlightFlags, const int reqsPerRank);
      void pollRequests();
      void checkInFlightRequests();
      void initCommVars();
      void destroyCommVars();

      void checkOutputStream();
      void outputStreamFinish(char * fmt, char * memory, char * printed, int & plen, int & maxLen);
      void outputStreamFlush(char * printed, int & plen, int & maxLen);
      void outputStreamWrite(const char * const s, const int start, const int end, char * printed, int & plen, int & maxLen);

      static void launchThread(void * param);
    public:
      GPUWorker(const int numSlots, const int numAsyncTrans, const int localDeviceID, const int gpuThreadIndex);
      ~GPUWorker();

      void setLocalID(const int id);
      void setGlobalID(const int id);
      void setPauseTime(const int ms);
      void start();

      void scheduleKernel(const GPUKernelFunction kernelFunc, const GPUCleanupFunction dtor, void * const param, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize = 0);
      void shutdown();

      void abort(const ErrorCode errorCode);

      bool isIdle();
      void waitForShutdown();

      static bool isGPUWorkerThread();
      static Target getRank();
  };
  extern std::vector<GPUWorker * > gpuWorkers;
}

#endif
