#include <dcgn/CPUWorker.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>
#include <stdexcept>
#include <cstdio>

namespace dcgn
{
  std::vector<CPUWorker * > cpuWorkers;
  ThreadSpecificData CPUWorker::workerHandle;

  const char * REQUEST_STRINGS[] =
  {
    "REQUEST_TYPE_NONE",
    "REQUEST_TYPE_SEND",                    // Send data from one thread to another (possibly non-local) thread.
    "REQUEST_TYPE_RECV",                    // Receive data sent from another (possibly non-local) thread.
    "REQUEST_TYPE_SEND_RECV",               // Send and recv data between threads with one call.
    "REQUEST_TYPE_SEND_RECV_REPLACE",       // Send and recv data between threads with one call.
    "REQUEST_TYPE_BARRIER",                 // Barrier with all the threads.
    "REQUEST_TYPE_BROADCAST",               // Barrier with all the threads.

    "REQUEST_TYPE_CPU_KERNEL",              // Run a kernel on a CPU.
    "REQUEST_TYPE_GPU_SETUP",               // Run a kernel on a GPU.
    "REQUEST_TYPE_GPU_KERNEL",              // Setup state and dynamic memory for a GPU kernel (must be done within the GPU thread).

    "REQUEST_TYPE_SHUTDOWN",                // Everything is done so shutdown.
  };

  void CPUWorker::checkError(const ErrorCode code)
  {
    if (code != DCGN_ERROR_NONE)
    {
      profiler->add("Error encountered - '%s' Exiting thread.", dcgn::getErrorString(code));
      dcgn::abort(code);
      errorHappened = true;
      throw std::logic_error("abort called.");
    }
  }
  void CPUWorker::loop()
  {
    bool done = false;
    while (!done && !errorHappened)
    {
      profiler->add("Work queue empty, going idle.");
      qCondVar.wait(qMutex);
      profiler->add("Received work, waking up.");
      while (!pendingReqs.empty() && !errorHappened)
      {
        Request * req = pendingReqs.front();
        pendingReqs.pop_front();
        servingRequest = true;
        qMutex.unlock();
        try
        {
          if (!errorHappened) serviceRequest(req, done);
          else                delete req;
        }
        catch (std::logic_error & e)
        {
          errorHappened = true;
        }
        qMutex.lock();
        servingRequest = false;
      }
    }
    qMutex.unlock();
  }
  void CPUWorker::addRequest(Request * const req)
  {
    // fprintf(stderr, "%s.%s.%d: cpu queue %d adding request of type %s\n", __FILE__, __FUNCTION__, __LINE__, cpuID, REQUEST_STRINGS[req->type]); fflush(stderr);
    qMutex.lock();
    if (pendingReqs.empty()) qCondVar.signal();
    pendingReqs.push_back(req);
    qMutex.unlock();
  }
  void CPUWorker::serviceRequest(Request * const req, bool & isShutdown)
  {
    // fprintf(stderr, "%s.%s.%d: cpu %d servicing request of type %s\n", __FILE__, __FUNCTION__, __LINE__, cpuID, REQUEST_STRINGS[req->type]); fflush(stderr);
    switch (req->type)
    {
    case REQUEST_TYPE_CPU_KERNEL:
      {
        profiler->add("Servicing kernel.");
        CPUKernelRequest * ckr = reinterpret_cast<CPUKernelRequest * >(req->request);
        ckr->func(ckr->param);
        delete ckr;
        profiler->add("Done servicing kernel.");
      }
      break;
    case REQUEST_TYPE_SHUTDOWN:
      profiler->add("Shutting down.");
      isShutdown = true;
      break;
    default:
      fprintf(stderr, "%s.%s.%d: Error, invalid request received. %d\n", __FILE__, __FUNCTION__, __LINE__, req->type);
      fflush(stderr);
      break;
    }
    delete req;
  }

  void CPUWorker::launchThread(void * param)
  {
    void ** params = reinterpret_cast<void ** >(param);
    CPUWorker * worker = reinterpret_cast<CPUWorker * >(params[0]);
    volatile bool * initialized = const_cast<volatile bool * >(reinterpret_cast<bool * >(params[1]));
    workerHandle.setValue(params[0]);

    profiler->setTitle("CPU Thread %d", worker->cpuID);

    worker->qMutex.lock();
    *initialized = true;
    worker->loop();
    profiler->add("Done looping.");
  }

  CPUWorker::CPUWorker(const int cpuThreadIndex)
    : cpuID(cpuThreadIndex), localID(cpuThreadIndex), servingRequest(false)
  {
  }
  CPUWorker::~CPUWorker()
  {
    outputStream.destroy();
  }
  void CPUWorker::setGlobalID(const int id)
  {
    globalID = id;
  }
  void CPUWorker::start()
  {
    volatile bool initialized = false;
    void * params[2] = { reinterpret_cast<void * >(this), reinterpret_cast<void * >(const_cast<bool * >(&initialized)) };

    errorHappened = false;
    outputStream.init();
    thread.start(launchThread, reinterpret_cast<void * >(params));
    while (!initialized) { }
  }

  void CPUWorker::scheduleKernel(const CPUKernelFunction kernelFunc, void * const param)
  {
    CPUKernelRequest * kernelReq = new CPUKernelRequest;
    Request * req = new Request;
    req->type = REQUEST_TYPE_CPU_KERNEL;
    req->request = reinterpret_cast<void * >(kernelReq);
    kernelReq->func = kernelFunc;
    kernelReq->param = param;
    addRequest(req);
  }
  void CPUWorker::shutdown()
  {
    Request * req = new Request;
    req->type = REQUEST_TYPE_SHUTDOWN;
    addRequest(req);
  }

  void CPUWorker::abort(const ErrorCode errorCode)
  {
    if (ioCondVar.getWaitingCount() > 0) return;  // the mpi thread will kill off this thread.
    if (!servingRequest)                          // the thread is idle, so signal an error and wake it up.
    {
      errorHappened = true;
      shutdown();
    }
  }

  bool CPUWorker::isIdle()
  {
    bool ret;
    qMutex.lock();
    ret = pendingReqs.empty() && !servingRequest;
    qMutex.unlock();
    return ret;
  }
  void CPUWorker::waitForShutdown()
  {
    thread.waitFor();
  }

  void CPUWorker::send(const Target dst, const void * const buffer, const int numBytes)
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());
    ErrorCode error;

    profiler->add("Initializing send (rank=%lld) of %d bytes (buf=%p) to %lld.", worker->globalID, numBytes, buffer, dst);
    worker->ioMutex.lock();
    mpiWorker->send(buffer, numBytes, worker->globalID, dst, 0, 0, worker->localID, worker->globalID, 0, &error, &worker->ioMutex, &worker->ioCondVar);
    worker->ioCondVar.wait(worker->ioMutex);
    profiler->add("Send completed.");
    worker->ioMutex.unlock();
    worker->checkError(error);
  }
  void CPUWorker::recv(const Target src, void * const buffer, const int maxBytes, CommStatus * const stat)
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());
    ErrorCode error;

    profiler->add("Initializing recv (rank=%lld) of %d bytes (buf=%p) from %lld.", worker->globalID, maxBytes, buffer, src);
    worker->ioMutex.lock();
    mpiWorker->recv(buffer, maxBytes, src, worker->globalID, stat, 0, worker->localID, worker->globalID, 0, &error, &worker->ioMutex, &worker->ioCondVar);
    worker->ioCondVar.wait(worker->ioMutex);
    profiler->add("Recv completed.");
    worker->ioMutex.unlock();
    worker->checkError(error);
  }
  void CPUWorker::sendRecvReplace(const Target dst, const Target src, void * const buffer, const int numBytes, CommStatus * const stat)
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());
    ErrorCode error;

    profiler->add("Initializing sendRecvReplace (rank=%lld) of %d bytes (buf=%p) from %lld, to %lld.", worker->globalID, numBytes, buffer, src, dst);
    worker->ioMutex.lock();
    mpiWorker->sendRecvReplace(buffer, numBytes, dst, src, stat, 0, worker->localID, worker->globalID, 0, &error, &worker->ioMutex, &worker->ioCondVar);
    worker->ioCondVar.wait(worker->ioMutex);
    profiler->add("SendRecvReplace completed.");
    worker->ioMutex.unlock();
    worker->checkError(error);
  }
  void CPUWorker::barrier()
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());
    ErrorCode error;

    profiler->add("Initializing barrier (rank=%lld).", worker->globalID);
    worker->ioMutex.lock();
    mpiWorker->barrier(worker->localID, worker->globalID, 0, &error, &worker->ioMutex, &worker->ioCondVar);
    worker->ioCondVar.wait(worker->ioMutex);
    profiler->add("Barrier completed.");
    worker->ioMutex.unlock();
    worker->checkError(error);
  }
  void CPUWorker::broadcast(const Target root, void * const bytes, const int numBytes)
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());
    ErrorCode error;

    profiler->add("Initializing broadcast (rank=%lld) of %d bytes (buf=%p) from root %lld.", worker->globalID, numBytes, bytes, root);
    worker->ioMutex.lock();
    mpiWorker->broadcast(root, bytes, numBytes, worker->localID, worker->globalID, 0, &error, &worker->ioMutex, &worker->ioCondVar);
    worker->ioCondVar.wait(worker->ioMutex);
    profiler->add("Broadcast completed.");
    worker->ioMutex.unlock();
    worker->checkError(error);
  }
  void CPUWorker::asyncSend(const Target dst, const void * const buffer, const int numBytes, AsyncRequest * const req)
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());

    profiler->add("Initializing asynchronous send (rank=%lld) of %d bytes (buf=%p) to %lld.", worker->globalID, numBytes, buffer, dst);
    if (!req) mpiWorker->send(buffer, numBytes, worker->globalID, dst, 0,           0,    worker->localID, worker->globalID, 0, 0, &worker->ioMutex, &worker->ioCondVar);
    else      mpiWorker->send(buffer, numBytes, worker->globalID, dst, &req->stat,  req,  worker->localID, worker->globalID, 0, 0, &worker->ioMutex, &worker->ioCondVar);
    profiler->add("Send completed.");
  }
  void CPUWorker::asyncRecv(const Target src,       void * const buffer, const int maxBytes, AsyncRequest * const req)
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());

    profiler->add("Initializing recv (rank=%lld) of %d bytes (buf=%p) from %lld.", worker->globalID, maxBytes, buffer, src);
    if (!req) mpiWorker->recv(buffer, maxBytes, src, worker->globalID, 0,           0,    worker->localID, worker->globalID, 0, 0, &worker->ioMutex, &worker->ioCondVar);
    else      mpiWorker->recv(buffer, maxBytes, src, worker->globalID, &req->stat,  req,  worker->localID, worker->globalID, 0, 0, &worker->ioMutex, &worker->ioCondVar);
    profiler->add("Recv completed.");
  }
  Target CPUWorker::getRank()
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());
    return worker->globalID;
  }
  bool CPUWorker::isCPUWorkerThread()
  {
    return workerHandle.getValue() != 0;
  }
  OutputStream & CPUWorker::output()
  {
    CPUWorker * const worker = reinterpret_cast<CPUWorker * >(workerHandle.getValue());
    return worker->outputStream;
  }
}
