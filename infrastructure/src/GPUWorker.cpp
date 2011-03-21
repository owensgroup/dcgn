#include <cuda_runtime.h>
#include <dcgn/GPUWorker.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <cuda_runtime_api.h>

static bool inDest = false;
#define checkCudaError(x)                                                                                                           \
{                                                                                                                                   \
  cudaError_t err = cudaGetLastError();                                                                                             \
  if (err != cudaSuccess)                                                                                                           \
  {                                                                                                                                 \
    inDest = true;                                                                                                                  \
    fprintf(stderr, "%s - Error on line %s.%s.%d - %s. Aborting.\n", x, __FILE__, __FUNCTION__, __LINE__, cudaGetErrorString(err)); \
    fflush(stderr);                                                                                                                 \
    exit(0);                                                                                                                        \
  }                                                                                                                                 \
}                                                                                                                                   \

namespace dcgn
{
  std::vector<GPUWorker * > gpuWorkers;
  ThreadSpecificData GPUWorker::workerHandle;
  void GPUWorker::checkError(const ErrorCode code)
  {
    if (code != DCGN_ERROR_NONE)
    {
      profiler->add("Error encountered - '%s' Exiting thread.", dcgn::getErrorString(code));
      errorHappened = true;
    }
  }
  void GPUWorker::loop()
  {
    bool done = false;
    while (!done && !errorHappened)
    {
      qMutex.unlock(); // unlock while servicing requests
      lastEvent = mpiWorker->wallTime();
      while (cudaStreamQuery(kernelStream) != cudaSuccess && !errorHappened)
      {
        if      (pause == 0) Thread::yield();
        else if (pause >  0) Thread::sleep(pause);
        // checkOutputStream();
        pollRequests();
        checkInFlightRequests();

        if (mpiWorker->wallTime() - lastEvent > static_cast<double>(timeout))
        {
          dcgn::abort(DCGN_ERROR_GPU_TIMEOUT);
        }
      }
      profiler->add("Done executing kernel");
      if (currentDtor)
      {
        currentDtor(currentParam);
        currentDtor = 0;
        currentParam = 0;
      }

      servingRequest = false;
      qMutex.lock();
      if (pendingReqs.empty() && !errorHappened)
      {
        qCondVar.wait(qMutex);
      }
      if (!pendingReqs.empty() && !errorHappened)
      {
        Request * req = pendingReqs.front();
        pendingReqs.pop_front();
        servingRequest = true;
        qMutex.unlock();
        serviceRequest(req, done);
        qMutex.lock();
      }
    }
    qMutex.unlock();
    servingRequest = false;
  }
  void GPUWorker::addRequest(Request * const req)
  {
    if (!thread.isRunning())
    {
      delete req;
      return;
    }
    qMutex.lock();
    if (pendingReqs.empty()) qCondVar.signal();
    pendingReqs.push_back(req);
    qMutex.unlock();
  }
  void GPUWorker::serviceRequest(Request * const req, bool & isShutdown)
  {
    switch (req->type)
    {
    case REQUEST_TYPE_GPU_KERNEL:
      {
        profiler->add("Servicing kernel.");
        GPUKernelRequest * gkr = reinterpret_cast<GPUKernelRequest * >(req->request);
        gkr->func(gkr->param, initReq, gkr->gridSize, gkr->blockSize, gkr->sharedMemSize, &kernelStream);
        currentDtor = gkr->dtor;
        currentParam = gkr->param;
        checkCudaError("kernelLaunch");
        delete gkr;
        delete req;
      }
      break;
    case REQUEST_TYPE_SHUTDOWN:
      profiler->add("Shutting down.");
      isShutdown = true;
      delete req;
      break;
    default:
      fprintf(stderr, "%s.%s.%d: Error, invalid request received. %d\n", __FILE__, __FUNCTION__, __LINE__, req->type);
      fflush(stderr);
      break;
    }
  }

  void GPUWorker::ensurePageLockedMemSize(const size_t size)
  {
    if (pageLockedMemSize < size)
    {
      if (pageLockedMem)
      {
        // profiler->add("freeing page locked buffer %p of size %d.\n", pageLockedMem, pageLockedMemSize);
        cudaFreeHost(pageLockedMem);                                          checkCudaError("freeHost");
      }
      if (pageLockedMemSize == 0) pageLockedMemSize = 1;
      while (pageLockedMemSize < size) pageLockedMemSize <<= 1;
      cudaMallocHost((void ** )&pageLockedMem, pageLockedMemSize);            checkCudaError("mallocHost");
      // profiler->add("allocated page locked buffer %p of size %d.", pageLockedMem, pageLockedMemSize);
    }
  }
  void GPUWorker::copyFromDevice(const bool pageLocked, void * const cpuMem, const void * const gpuMem, const size_t size)
  {
    void * dst = cpuMem;
    if (!pageLocked)
    {
      ensurePageLockedMemSize(size);
      dst = pageLockedMem;
    }
    cudaMemcpyAsync(dst, gpuMem, size, cudaMemcpyDeviceToHost, memcpyStream); checkCudaError("memcpyAsync");
    cudaStreamSynchronize(memcpyStream);                                      checkCudaError("streamSynchronize");
    if (!pageLocked)
    {
      memcpy(cpuMem, dst, size);
    }
  }
  void GPUWorker::copyToDevice  (const bool pageLocked, void * const gpuMem, const void * const cpuMem, const size_t size)
  {
    void * src = const_cast<void * >(cpuMem);
    if (!pageLocked)
    {
      ensurePageLockedMemSize(size);
      src = pageLockedMem;
      memcpy(src, cpuMem, size);
    }
    cudaMemcpyAsync(gpuMem, src, size, cudaMemcpyHostToDevice, memcpyStream); checkCudaError("memcpyAsync");
    cudaStreamSynchronize(memcpyStream);                                      checkCudaError("streamSynchronize");
  }

  void * GPUWorker::findPageLockedBuffer(const int size)
  {
    std::pair<void * , int> plBuf(0, -1);

    if (size)
    {
      // first go through and try to find a suitable buffer.
      for (std::map<void * , int>::iterator it = pageLockedBufs.begin(); it != pageLockedBufs.end(); ++it)
      {
        if (it->second >= size)
        {
          plBuf = *it;
          pageLockedBufs.erase(it);
          // profiler->add("found page locked buffer %p of size %d for request of size %d.", plBuf.first, plBuf.second, size);
          break;
        }
      }
      // if we can't find one, find the largest free buffer, delete it, and reallocate it.
      if (plBuf.second == -1)
      {
        for (std::map<void * , int>::iterator it = pageLockedBufs.begin(); it != pageLockedBufs.end(); ++it)
        {
          if (it->second > plBuf.second)
          {
            plBuf = *it;
          }
        }
        // we found a suitable buffer. delete it, so we can add another buffer in.
        if (plBuf.second > 0)
        {
          // profiler->add("deleting page locked buffer %p of size %d.", plBuf.first, plBuf.second);
          pageLockedBufs.erase(pageLockedBufs.find(plBuf.first));
          cudaFreeHost(plBuf.first);
        }
        int tempSize = 1024;
        while (tempSize < size) tempSize <<= 1;
        cudaMallocHost((void ** )&plBuf.first, tempSize); checkCudaError("mallocHost");
        plBuf.second = tempSize;
        // profiler->add("allocated page locked buffer %p of size %d for request of size %d.", plBuf.first, plBuf.second, size);
      }
      usedBufs[plBuf.first] = plBuf.second;
      memset(plBuf.first, 0, size);
      return plBuf.first;
    }
    profiler->add("failed to find buffer for some reason or another.");
    return 0;
  }
  void   GPUWorker::releasePageLockedBuffer(void * const buffer)
  {
    pageLockedBufs[buffer] = usedBufs[buffer];
    usedBufs.erase(buffer);
  }
  void GPUWorker::copyUpAndInit(InternalRequest * const req, GPUIORequest * const localReq)
  {
    req->buf = findPageLockedBuffer(localReq->numBytes);
    profiler->add("Copying %d bytes of data from device for send.", localReq->numBytes);
    copyFromDevice(true, req->buf, localReq->buf, localReq->numBytes);
  }
  void GPUWorker::pollRequestsSub(std::vector<GPUIORequest> & reqs, GPUIORequest * gpuReqs, std::vector<int> & inFlightFlags, const int reqsPerRank)
  {
    copyFromDevice(false, &reqs[0], gpuReqs, sizeof(GPUIORequest) * (int)reqs.size());
    for (int i = 0; i < (int)reqs.size(); ++i)
    {
      if (!inFlightFlags[i]) // did we already begin to service this request?
      {
        // fprintf(stderr, "localReqs[%d].type: %s\n", i, REQUEST_STRINGS[localReqs[i].type]); fflush(stderr);
        if (reqs[i].type != REQUEST_TYPE_NONE)
        {
          lastEvent = mpiWorker->wallTime();
        }
        InternalRequest * req = 0;
        if (reqs[i].type != REQUEST_TYPE_NONE && reqs[i].type != REQUEST_TYPE_MALLOC && reqs[i].type != REQUEST_TYPE_FREE)
        {
          req = new InternalRequest;
          req->done = false;
          req->rank = globalID + i / reqsPerRank;
          req->type = reqs[i].type;
          req->gpuReq = gpuReqs + i;
          req->localReq = &reqs[i];
          req->inFlightFlag = &inFlightFlags[i];
          *req->inFlightFlag = true;
          inFlightReqs.push_back(req);
          req->done = false;
        }
        switch (reqs[i].type)
        {
        case REQUEST_TYPE_NONE:
          break;
        case REQUEST_TYPE_SEND:
          copyUpAndInit(req, &reqs[i]);
          profiler->add("Initializing send (rank=%lld) of %d bytes (buf=%p) to %lld.", globalID + i / reqsPerRank, reqs[i].numBytes, req->buf, reqs[i].to);
          mpiWorker->send(req->buf, reqs[i].numBytes, globalID + i / reqsPerRank, reqs[i].to, 0, 0, localID, globalID + i / reqsPerRank, &req->done, &req->error, 0, 0);
          break;
        case REQUEST_TYPE_RECV:
          req->buf = findPageLockedBuffer(reqs[i].numBytes);
          profiler->add("Initializing recv (rank=%lld) of %d bytes (buf=%p) from %lld.", globalID + i / reqsPerRank, reqs[i].numBytes, req->buf, reqs[i].from);
          mpiWorker->recv(req->buf, reqs[i].numBytes, reqs[i].from, globalID + i / reqsPerRank, &reqs[i].req.stat, 0, localID, globalID + i / reqsPerRank, &req->done, &req->error, 0, 0);
          break;
        case REQUEST_TYPE_SEND_RECV_REPLACE:
          copyUpAndInit(req, &reqs[i]);
          profiler->add("Initializing sendRecvReplace (rank=%lld) of %d bytes (buf=%p) from %lld, to %lld.", globalID + i / reqsPerRank, reqs[i].numBytes, req->buf, reqs[i].from, reqs[i].to);
          mpiWorker->sendRecvReplace(req->buf, reqs[i].numBytes, reqs[i].to, reqs[i].from, &reqs[i].req.stat, 0, localID, globalID + i / reqsPerRank, &req->done, &req->error, 0, 0);
          break;
        case REQUEST_TYPE_BARRIER:
          req->buf = 0;
          profiler->add("Initializing barrier (rank=%lld).", globalID + i / reqsPerRank);
          mpiWorker->barrier(localID, globalID + i / reqsPerRank, &req->done, &req->error, 0, 0);
          break;
        case REQUEST_TYPE_BROADCAST:
          req->buf = findPageLockedBuffer(reqs[i].numBytes);
          if (reqs[i].from == globalID + i / reqsPerRank)
          {
            copyFromDevice(true, req->buf, reqs[i].buf, reqs[i].numBytes);
          }
          profiler->add("Initializing broadcast (rank=%lld) of %d bytes (buf=%p) from root %lld.", globalID + i / reqsPerRank, reqs[i].numBytes, req->buf, reqs[i].from);
          mpiWorker->broadcast(reqs[i].from, req->buf, reqs[i].numBytes, localID, globalID + i / reqsPerRank, &req->done, &req->error, 0, 0);
          break;
        case REQUEST_TYPE_MALLOC:
          {
            GPUIORequest ioreq;
            ioreq.type = REQUEST_TYPE_NONE;
            ioreq.done = true;

            cudaMalloc(&ioreq.buf, reqs[i].numBytes);
            copyToDevice(false, gpuReqs + i, &ioreq, sizeof(GPUIORequest));
          }
          break;
        case REQUEST_TYPE_FREE:
          {
            GPUIORequest ioreq;
            ioreq.type = REQUEST_TYPE_NONE;
            ioreq.done = true;

            cudaFree(reqs[i].buf);
            copyToDevice(false, gpuReqs + i, &ioreq, sizeof(GPUIORequest));
          }
          break;
        default:
          fprintf(stderr, "%s.%s.%d: Error, invalid request type found.\n", __FILE__, __FUNCTION__, __LINE__);
          fflush(stderr);
          exit(0);
          break;
        }
      }
    }
  }
  void GPUWorker::pollRequests()
  {
    pollRequestsSub(localReqs,      initReq.gpuReqs,      inFlight,       1);
    pollRequestsSub(localAsyncReqs, initReq.gpuAsyncReqs, inFlightAsync,  asyncCount);
#if 0
    {
      unsigned long long * dbg = new unsigned long long[1024];
      copyFromDevice(false, dbg, initReq.debugInfo, sizeof(unsigned long long) * 1024);
      char buf[1024] = "";
      for (int i = 0; i < 15; ++i)
      {
        char c[40];
        sprintf(c, "0x%010llx ", dbg[i]);
        strcat(buf, c);
      }
      profiler->add("{ %s}", buf);
      delete [] dbg;
    }
#endif
  }
  void GPUWorker::checkInFlightRequests()
  {
    for (int i = 0; i < (int)inFlightReqs.size(); )
    {
      if (inFlightReqs[i]->done)
      {
        checkError(inFlightReqs[i]->error);
        switch (inFlightReqs[i]->localReq->type)
        {
        case REQUEST_TYPE_SEND:
          profiler->add("Send completed (rank=%lld).", inFlightReqs[i]->rank);
          break;
        case REQUEST_TYPE_RECV:
          profiler->add("Recv completed (rank=%lld).", inFlightReqs[i]->rank);
          profiler->add("Copying data to device.");
          copyToDevice(true,  inFlightReqs[i]->localReq->buf,     inFlightReqs[i]->buf,                 inFlightReqs[i]->localReq->numBytes);
          copyToDevice(false, &inFlightReqs[i]->gpuReq->req.stat, &inFlightReqs[i]->localReq->req.stat, sizeof(CommStatus));
          profiler->add("Done copying data and status back to device.");
          break;
        case REQUEST_TYPE_SEND_RECV_REPLACE:
          profiler->add("SendRecvReplace completed (rank=%lld).", inFlightReqs[i]->rank);
          copyToDevice(true,  inFlightReqs[i]->localReq->buf, inFlightReqs[i]->buf, inFlightReqs[i]->localReq->req.stat.numBytes);
          profiler->add("Done copying data and status back to device.");
          break;
        case REQUEST_TYPE_BARRIER:
          profiler->add("Barrier completed (rank=%lld).", inFlightReqs[i]->rank);
          break;
        case REQUEST_TYPE_BROADCAST:
          profiler->add("Broadcast completed (rank=%lld, from=%d).", inFlightReqs[i]->rank, inFlightReqs[i]->localReq->from);
          if (inFlightReqs[i]->localReq->from != inFlightReqs[i]->rank) // this means we are *not* the root and thus *received* data.
          {
            profiler->add("Copying data to device.");
            copyToDevice(true,  inFlightReqs[i]->localReq->buf, inFlightReqs[i]->buf, inFlightReqs[i]->localReq->numBytes);
            profiler->add("Done copying data and status back to device.");
          }
          break;
        default:
          fprintf(stderr, "%s.%s.%d: Error, invalid request type found.\n", __FILE__, __FUNCTION__, __LINE__);
          fflush(stderr);
          exit(0);
          break;
        }
        if (inFlightReqs[i]->buf)
        {
          releasePageLockedBuffer(inFlightReqs[i]->buf);
        }
        inFlightReqs[i]->localReq->req.completed = true;
        inFlightReqs[i]->localReq->done = true;
        inFlightReqs[i]->localReq->type = REQUEST_TYPE_NONE;
        profiler->add("Telling GPU to continue execution.");
        copyToDevice(false, inFlightReqs[i]->gpuReq, inFlightReqs[i]->localReq, sizeof(GPUIORequest));
        profiler->add("Done.");
        *inFlightReqs[i]->inFlightFlag = false;
        delete inFlightReqs[i];
        inFlightReqs.erase(inFlightReqs.begin() + i);
      }
      else
      {
        ++i;
      }
    }
  }
  void GPUWorker::initCommVars()
  {
    int cpuFlag = 0;
    pageLockedMemSize = 0;
    pageLockedMem = 0;
    localReqs.      resize(slots);
    localAsyncReqs. resize(slots * asyncCount);
    inFlight.       resize(slots, false);
    inFlightAsync.  resize(slots * asyncCount, false);
    profiler->add("Setting GPU device to %d.", deviceID);
    cudaSetDevice(deviceID);
    profiler->add("Checking for errors.");
    checkCudaError("setDevice");
    profiler->add("Allocating space on the GPU.");
    cudaMalloc(reinterpret_cast<void ** >(&initReq.gpuReqs),        sizeof(GPUIORequest) * slots);                  checkCudaError("malloc");
    cudaMalloc(reinterpret_cast<void ** >(&gpuStream),              sizeof(OutputStream));                          checkCudaError("malloc");
    cudaMalloc(reinterpret_cast<void ** >(&outputStream.memory),    sizeof(char) * 8192);                           checkCudaError("malloc");
    cudaMalloc(reinterpret_cast<void ** >(&outputStream.cpuFlag),   sizeof(int));                                   checkCudaError("malloc");

    if (asyncCount > 0)
    {
      cudaMalloc(reinterpret_cast<void ** >(&initReq.gpuAsyncReqs),   sizeof(GPUIORequest) * slots * asyncCount);     checkCudaError("malloc");
      cudaMalloc(reinterpret_cast<void ** >(&initReq.localAsyncReqs), sizeof(AsyncRequest * ) * slots * asyncCount);  checkCudaError("malloc");
      AsyncRequest ** asyncs = new AsyncRequest*[slots * asyncCount];
      for (int i = 0; i < slots * asyncCount; ++i) asyncs[i] = reinterpret_cast<AsyncRequest * >(0xDeadBeef);
      cudaMemcpy(initReq.localAsyncReqs, asyncs, sizeof(AsyncRequest * ) * slots * asyncCount, cudaMemcpyHostToDevice);
      delete [] asyncs;
    }

    initReq.gpuNodeID = mpiWorker->getMPIRank();
    initReq.gpuSize = mpiWorker->getGlobalSize();
    initReq.gpuRank = globalID;
    initReq.numAsyncReqs = asyncCount;
    {
      int * gpu, * cpu;
      unsigned long long * dbg;
      cudaMalloc(reinterpret_cast<void ** >(&initReq.cpuRanks),   sizeof(int) * mpiWorker->getGlobalCPUCount());
      cudaMalloc(reinterpret_cast<void ** >(&initReq.gpuRanks),   sizeof(int) * mpiWorker->getGlobalGPUCount());
      cudaMalloc(reinterpret_cast<void ** >(&initReq.debugInfo),  sizeof(unsigned long long) * 1024);
      cpu  = new int[mpiWorker->getGlobalCPUCount()];
      gpu  = new int[mpiWorker->getGlobalGPUCount()];
      dbg  = new unsigned long long[1024];
      for (int i = mpiWorker->getGlobalCPUCount() - 1; i >= 0; --i) { cpu[i] = mpiWorker->getTargetForCPU(i);    }
      for (int i = mpiWorker->getGlobalGPUCount() - 1; i >= 0; --i) { gpu[i] = mpiWorker->getTargetForGPU(i, 0); }
      for (int i = 0; i < 1024; ++i) { dbg[i] = 0x200; }
      cudaMemcpy(initReq.cpuRanks,  cpu, sizeof(int) * mpiWorker->getGlobalCPUCount(),  cudaMemcpyHostToDevice);
      cudaMemcpy(initReq.gpuRanks,  gpu, sizeof(int) * mpiWorker->getGlobalGPUCount(),  cudaMemcpyHostToDevice);
      cudaMemcpy(initReq.debugInfo, dbg, sizeof(unsigned long long) * 1024,             cudaMemcpyHostToDevice);
      delete [] cpu;
      delete [] gpu;
      delete [] dbg;
    }
    initReq.outputStream = gpuStream;

    profiler->add("Memsetting on the GPU.");
    if (asyncCount > 0)
    {
      cudaMemset(initReq.gpuAsyncReqs,    0, sizeof(GPUIORequest)     * slots * asyncCount);              checkCudaError("memset");
    }
    cudaMemset(initReq.gpuReqs, 0, sizeof(GPUIORequest) * slots);                                       checkCudaError("memset");
    cudaStreamCreate(&kernelStream);                                                                    checkCudaError("streamCreate");
    cudaStreamCreate(&memcpyStream);                                                                    checkCudaError("streamCreate");

    profiler->add("Memcpying to the GPU.");
    cudaMemcpy(outputStream.cpuFlag, &cpuFlag,         sizeof(int),          cudaMemcpyHostToDevice);   checkCudaError("memcpy");
    cudaMemcpy(gpuStream,            &outputStream,    sizeof(OutputStream), cudaMemcpyHostToDevice);   checkCudaError("memcpy");
    profiler->add("Done initializing communication variables.");
  }
  void GPUWorker::destroyCommVars()
  {
#if 0
    {
      unsigned long long * dbg = new unsigned long long[1024];
      copyFromDevice(false, dbg, initReq.debugInfo, sizeof(unsigned long long) * 1024);
      char buf[1024] = "";
      for (int i = 0; i < 22; ++i)
      {
        char c[40];
        sprintf(c, "0x%010llx ", dbg[i]);
        strcat(buf, c);
      }
      profiler->add("{ %s}", buf);
      delete [] dbg;

      AsyncRequest ** dbg2 = new AsyncRequest*[slots * asyncCount];
      copyFromDevice(false, dbg2, initReq.localAsyncReqs, sizeof(AsyncRequest * ) * slots * asyncCount);

      buf[0] = 0;
      for (int i = 0; i < slots * asyncCount; ++i)
      {
        char c[40];
        sprintf(c, "%p ", dbg2[i]);
        strcat(buf, c);
      }
      profiler->add("{ %s}", buf);
    }
#endif
    cudaFree(initReq.gpuReqs);        checkCudaError("free");
    if (asyncCount > 0)
    {
      cudaFree(initReq.gpuAsyncReqs);   checkCudaError("free");
      cudaFree(initReq.localAsyncReqs); checkCudaError("free");
    }
    cudaFree(initReq.cpuRanks);       checkCudaError("free");
    cudaFree(initReq.gpuRanks);       checkCudaError("free");
    cudaStreamDestroy(kernelStream);  checkCudaError("streamDestroy");
    cudaStreamDestroy(memcpyStream);  checkCudaError("streamDestroy");
    cudaFree(outputStream.memory);    checkCudaError("free");
    cudaFree(outputStream.cpuFlag);   checkCudaError("free");
    if (pageLockedMemSize > 0)
    {
      cudaFreeHost(pageLockedMem); checkCudaError("freeHost");
      for (std::map<void * , int>::iterator it = pageLockedBufs.begin();  it != pageLockedBufs.end(); ++it) cudaFreeHost(it->first);
      for (std::map<void * , int>::iterator it = usedBufs.begin();        it != usedBufs.end();       ++it) cudaFreeHost(it->first);
    }
  }

  void GPUWorker::launchThread(void * param)
  {
    void ** params = reinterpret_cast<void ** >(param);
    GPUWorker * worker = reinterpret_cast<GPUWorker * >(params[0]);
    volatile bool * initialized = const_cast<volatile bool * >(reinterpret_cast<bool * >(params[1]));
    workerHandle.setValue(params[0]);

    profiler->setTitle("GPU Thread %d", worker->gpuID);
    profiler->add("Initializing GPU variables.");
    worker->initCommVars();
    worker->qMutex.lock();
    *initialized = true;
    worker->timeout = 2000000000;
    worker->loop();
    worker->destroyCommVars();
    profiler->add("Done looping.");
  }

  void GPUWorker::checkOutputStream()
  {
    int cpuFlag;
    copyFromDevice(false, &cpuFlag, outputStream.cpuFlag, sizeof(int));
    if (cpuFlag == 1)
    {
      char * fmt, * memory, * printed;
      int plen, maxLen;

      maxLen = 8192;
      plen = 0;
      printed = new char[maxLen];

      copyFromDevice(false, &outputStream, gpuStream, sizeof(OutputStream));
      fmt = new char[outputStream.fmtLen + 1];
      memory = new char[outputStream.memLen];
      copyFromDevice(false, fmt,    outputStream.fmt,    outputStream.fmtLen + 1);
      copyFromDevice(false, memory, outputStream.memory, outputStream.memLen);

      outputStreamFinish(fmt, memory, printed, plen, maxLen);
      outputStreamFlush(printed, plen, maxLen);

      delete [] fmt;
      delete [] memory;
      delete [] printed;
      cpuFlag = 0;
      copyToDevice(false, outputStream.cpuFlag, &cpuFlag, sizeof(int));
    }
  }
  void GPUWorker::outputStreamFinish(char * fmt, char * memory, char * printed, int & plen, int & maxLen)
  {
    char * ptr = memory;
    char formatString[1024], formatted[1024];
    int lastByte = 0, i;
    bool done;
    ptr = reinterpret_cast<char * >(memory);
    for (i = 0; fmt[i]; )
    {
      if (fmt[i] == '%')
      {
        int width = -1, prec = -1;
        int percLoc = i;

        bool foundDot = false, foundWidth = false, foundPrec = false;
        outputStreamWrite(fmt, lastByte, i, printed, plen, maxLen);
        done = false;
        ++i;
        while (!done)
        {
          switch (fmt[i])
          {
          case '.':
            foundDot = true;
            ++i;
            break;
          case '*':
            if (!foundDot)
            {
              width = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(int);
              foundWidth = true;
            }
            else
            {
              prec = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(int);
              foundPrec = true;
            }
            ++i;
            break;
          case 0:
            outputStreamWrite(fmt, percLoc, i, printed, plen, maxLen);
            done = true;
            break;
          case 'c':
            {
              int ival = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(int);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, ival);
              else if (foundWidth)              sprintf(formatted, formatString, width,       ival);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, ival);
              else                              sprintf(formatted, formatString,              ival);
              outputStreamWrite(formatted, 0, strlen(formatted), printed, plen, maxLen);
              ++i;
              done = true;
            }
            break;
          case 'd':
          case 'i':
          case 'o':
          case 'u':
          case 'x':
          case 'X':
            if (i - percLoc > 3 && fmt[i - 1] == 'l' && fmt[i - 2] == 'l')
            {
              long long lval = *reinterpret_cast<long long * >(ptr);
              ptr += sizeof(long long);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, lval);
              else if (foundWidth)              sprintf(formatted, formatString, width,       lval);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, lval);
              else                              sprintf(formatted, formatString,              lval);
              outputStreamWrite(formatted, 0, strlen(formatted), printed, plen, maxLen);
              ++i;
              done = true;
            }
            else
            {
              int ival = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(int);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, ival);
              else if (foundWidth)              sprintf(formatted, formatString, width,       ival);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, ival);
              else                              sprintf(formatted, formatString,              ival);
              outputStreamWrite(formatted, 0, strlen(formatted), printed, plen, maxLen);
              ++i;
              done = true;
            }
            break;
          case 'n':
            ptr += sizeof(int * );
            ++i;
            done = true;
            break;
          case 'e':
          case 'E':
          case 'f':
          case 'g':
          case 'G':
            if (i - percLoc > 2 && fmt[i - 1] == 'L')
            {
              long double dval = *reinterpret_cast<long double * >(ptr);
              ptr += sizeof(long double);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, dval);
              else if (foundWidth)              sprintf(formatted, formatString, width,       dval);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, dval);
              else                              sprintf(formatted, formatString,              dval);
              outputStreamWrite(formatted, 0, strlen(formatted), printed, plen, maxLen);
              ++i;
              done = true;
            }
            else
            {
              double dval = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(double);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, dval);
              else if (foundWidth)              sprintf(formatted, formatString, width,       dval);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, dval);
              else                              sprintf(formatted, formatString,              dval);
              outputStreamWrite(formatted, 0, strlen(formatted), printed, plen, maxLen);
              ++i;
              done = true;
            }
            break;
          case 's':
          case 'p':
            {
              const char * sval = *reinterpret_cast<const char ** >(ptr);
              ptr += sizeof(char * );
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, sval);
              else if (foundWidth)              sprintf(formatted, formatString, width,       sval);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, sval);
              else                              sprintf(formatted, formatString,              sval);
              outputStreamWrite(formatted, 0, strlen(formatted), printed, plen, maxLen);
              ++i;
              done = true;
            }
            break;
          default:
            ++i;
            break;
          }
        }
        lastByte = i;
      }
      else
      {
        ++i;
      }
    }
    outputStreamWrite(fmt, lastByte, i, printed, plen, maxLen);
    ptr = reinterpret_cast<char * >(memory);
    outputStreamFlush(printed, plen, maxLen);
  }
  void GPUWorker::outputStreamFlush(char * printed, int & plen, int & maxLen)
  {
    if (plen > 0)
    {
      fwrite(printed, plen, 1, stderr);
      fflush(stderr);
      plen = 0;
    }
  }
  void GPUWorker::outputStreamWrite(const char * const s, const int start, const int end, char * printed, int & plen, int & maxLen)
  {
    if (end - start + plen > maxLen)
    {
      outputStreamFlush(printed, plen, maxLen);
    }
    memcpy(printed + plen, s + start, end - start);
    plen += end - start;
  }
  GPUWorker::GPUWorker(const int numSlots, const int numAsyncTrans, const int localDeviceID, const int gpuThreadIndex)
   : slots(numSlots), asyncCount(numAsyncTrans), deviceID(localDeviceID), gpuID(gpuThreadIndex), servingRequest(false), currentDtor(0), currentParam(0)
  {
  }
  GPUWorker::~GPUWorker()
  {
  }

  void GPUWorker::setLocalID(const int id)
  {
    localID = id;
  }
  void GPUWorker::setGlobalID(const int id)
  {
    globalID = id;
  }
  void GPUWorker::setPauseTime(const int ms)
  {
    pause = ms;
  }
  void GPUWorker::start()
  {
    errorHappened = false;
    bool initialized = false;
    void * params[2] = { reinterpret_cast<void * >(this), reinterpret_cast<void * >(const_cast<bool * >(&initialized)) };
    thread.start(launchThread, reinterpret_cast<void * >(params));
    while (!initialized) { Thread::yield(); }
  }

  void GPUWorker::scheduleKernel(const GPUKernelFunction kernelFunc, const GPUCleanupFunction dtor, void * const param, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize)
  {
    Request * req = new Request;
    GPUKernelRequest * gkr = new GPUKernelRequest;

    req->type = REQUEST_TYPE_GPU_KERNEL;
    req->request = reinterpret_cast<void * >(gkr);
    gkr->func = kernelFunc;
    gkr->dtor = dtor;
    gkr->param = param;
    gkr->blockSize = blockSize;
    gkr->gridSize = gridSize;
    gkr->sharedMemSize = sharedMemSize;
    addRequest(req);
  }
  void GPUWorker::shutdown()
  {
    Request * req = new Request;
    req->type = REQUEST_TYPE_SHUTDOWN;
    addRequest(req);
  }

  void GPUWorker::abort(const ErrorCode errorCode)
  {
    errorHappened = true;
    shutdown();
  }

  bool GPUWorker::isIdle()
  {
    bool ret;
    qMutex.lock();
    ret = pendingReqs.empty() && !servingRequest;
    qMutex.unlock();
    return ret;
  }
  void GPUWorker::waitForShutdown()
  {
    thread.waitFor();
  }

  bool GPUWorker::isGPUWorkerThread()
  {
    return workerHandle.getValue() != 0;
  }
  Target GPUWorker::getRank()
  {
    GPUWorker * const worker = reinterpret_cast<GPUWorker * >(workerHandle.getValue());
    return worker->globalID;
  }
}
