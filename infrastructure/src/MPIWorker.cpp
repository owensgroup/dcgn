#define MPICX_SKIP_MPICXX
#include <mpi.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/BarrierRequest.h>
#include <dcgn/BroadcastRequest.h>
#include <dcgn/Profiler.h>
#include <dcgn/RecvRequest.h>
#include <dcgn/SendRequest.h>
#include <dcgn/SendRecvRequest.h>
#include <dcgn/ShutdownRequest.h>
#include <dcgn/Thread.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#define CHECK_MPI(x)                                                                                \
{                                                                                                   \
  int err = (x);                                                                                    \
  if (err != MPI_SUCCESS)                                                                           \
  {                                                                                                 \
    fprintf(stderr, "%s.%s.%d: MPI_Function failed - %s.\n", __FILE__, __FUNCTION__, __LINE__, #x); \
    fflush(stderr);                                                                                 \
    MPI_Abort(MPI_COMM_WORLD, 0);                                                                   \
  }                                                                                                 \
}                                                                                                   \

#include <cstdio>
#include <typeinfo>

namespace dcgn
{
  MPIWorker * mpiWorker = 0;

  void MPIWorker::killRequests(const ErrorCode errorCode)
  {
    profiler->add("Killing all events.");
    for (int i = 0; i < (int)inFlightReqs.size(); ++i)
    {
      inFlightReqs[i]->setError(errorCode);
      delete inFlightReqs[i];
    }
    for (std::list<IORequest * >::iterator it = pendingReqs.begin(); it != pendingReqs.end(); ++it)
    {
      (*it)->setError(errorCode);
      delete *it;
    }
    inFlightReqs.clear();
    pendingReqs.clear();
    errorHappened = true;
  }
  void MPIWorker::loop()
  {
    bool done = false;
    while (!done && !errorHappened)
    {
      if (pendingReqs.empty())
      {
        qCondVar.wait(qMutex);
      }
      profiler->add("Received work, waking up.");
      while ((!pendingReqs.empty() || !inFlightReqs.empty()) && !errorHappened)
      {
        IORequest * req = 0;

        if (MPI_Wtime() - lastEvent > static_cast<double>(timeout))
        {
          qMutex.unlock();
          dcgn::abort(DCGN_ERROR_MPI_TIMEOUT);
          qMutex.lock();
        }

        if (!pendingReqs.empty() && !errorHappened)
        {
          req = pendingReqs.front();
          pendingReqs.pop_front();
        }
        qMutex.unlock();
        if (req && !errorHappened)
        {
          lastEvent = MPI_Wtime();
          serviceRequest(req, done);
          req = pendingReqs.front();
        }
        checkInFlightReqs();
        if (!req && !errorHappened)
        {
          if      (pause == 0) Thread::yield();
          else if (pause >  0) Thread::sleep(pause);
        }
      }
      profiler->add("Work queue empty, going idle.");
      qMutex.lock();
    }
    qMutex.unlock();
    if (errorHappened)
    {
      killRequests(DCGN_ERROR_MPI);
    }
  }
  void MPIWorker::addRequest(IORequest * const req)
  {
    if (!thread.isRunning() || errorHappened)
    {
      req->setError(DCGN_ERROR_MPI);
      delete req;
      return;
    }
    qMutex.lock();
    if (pendingReqs.empty()) qCondVar.signal();
    pendingReqs.push_back(req);
    qMutex.unlock();
  }
  void MPIWorker::serviceRequest(IORequest * const req, bool & isShutdown)
  {
    req->profileStart();
    if (req->getType() == REQUEST_TYPE_SHUTDOWN)
    {
      isShutdown = true;
      delete req;
    }
    else if (req->poll(inFlightReqs))
    {
      lastEvent = MPI_Wtime();
      delete req;
    }
    else
    {
      inFlightReqs.push_back(req);
    }
  }


  void MPIWorker::createTargetMapping()
  {
    int t[3];
    if (id == 0)
    {
      int curRank = 0;
      rankStart = 0;
      for (int i = 0; i < cpus; ++i)
      {
        mpiRanks.push_back(0);
        cpuRanks.push_back(curRank++);
      }
      for (int i = 0; i < gpus; ++i)
      {
        gpuRanks.push_back(curRank);
        curRank += slots;
        for (int j = 0; j < slots; ++j)
        {
          mpiRanks.push_back(0);
        }
      }
      for (int i = 1; i < size; ++i)
      {
        int arr[3];
        MPI_Status stat;
        CHECK_MPI(MPI_Send(&curRank, 1, MPI_INT, i, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Recv(arr, 3, MPI_INT, i, 0, MPI_COMM_WORLD, &stat));
        for (int j = 0; j < arr[0]; ++j)
        {
          mpiRanks.push_back(i);
          cpuRanks.push_back(curRank++);
        }
        for (int j = 0; j < arr[1]; ++j)
        {
          gpuRanks.push_back(curRank);
          curRank += arr[2];
          for (int k = 0; k < arr[2]; ++k)
          {
            mpiRanks.push_back(i);
          }
        }
      }
      t[0] = (int)mpiRanks.size();
      t[1] = (int)cpuRanks.size();
      t[2] = (int)gpuRanks.size();
      CHECK_MPI(MPI_Bcast(t, 3, MPI_INT, 0, MPI_COMM_WORLD));
    }
    else
    {
      int arr[3] = { cpus, gpus, slots };
      MPI_Status stat;
      CHECK_MPI(MPI_Recv(&rankStart, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat));
      CHECK_MPI(MPI_Send(arr, 3, MPI_INT, 0, 0, MPI_COMM_WORLD));
      CHECK_MPI(MPI_Bcast(t, 3, MPI_INT, 0, MPI_COMM_WORLD));
      mpiRanks.resize(t[0], -1);
      cpuRanks.resize(t[1], -1);
      gpuRanks.resize(t[2], -1);
    }
    CHECK_MPI(MPI_Bcast(&mpiRanks[0], t[0], MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&cpuRanks[0], t[1], MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&gpuRanks[0], t[2], MPI_INT, 0, MPI_COMM_WORLD));
  }
  void MPIWorker::checkInFlightReqs()
  {
    for (int i = 0; i < (int)inFlightReqs.size(); )
    {
      if (inFlightReqs[i]->poll(inFlightReqs))
      {
        delete inFlightReqs[i];
        inFlightReqs.erase(inFlightReqs.begin() + i);
        lastEvent = MPI_Wtime();
      }
      else ++i;
    }
  }

  void MPIWorker::launchThread(void * param)
  {
    void ** params = reinterpret_cast<void ** >(param);
    MPIWorker * worker = reinterpret_cast<MPIWorker * >(params[0]);
    bool * initialized = reinterpret_cast<bool * >(params[1]);

    // MPI_Init(argc, argv);

    profiler->setTitle("MPI Thread");
    worker->createTargetMapping();

    worker->qMutex.lock();
    *initialized = true;
    worker->timeout = 2000000000;
    worker->lastEvent = MPI_Wtime();
    worker->loop();
    profiler->add("Done looping.");
    // MPI_Finalize();
  }

  MPIWorker::MPIWorker(int * argc, char *** argv) : cpus(0), gpus(0), slots(0), pause(-1)
  {
    CHECK_MPI(MPI_Init(argc, argv));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &id));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  }
  MPIWorker::~MPIWorker()
  {
    CHECK_MPI(MPI_Finalize());
  }

  void MPIWorker::setCPUInfo(const int totalLocalCPUs)
  {
    cpus = totalLocalCPUs;
  }
  void MPIWorker::setGPUInfo(const int totalLocalGPUs, const int slotsPerGPU)
  {
    gpus = totalLocalGPUs;
    slots = slotsPerGPU;
  }
  void MPIWorker::setPauseTime(const int ms)
  {
    pause = ms;
  }
  void MPIWorker::start()
  {
    bool initialized = false;
    void * params[] =
    {
      reinterpret_cast<void * >(this),
      reinterpret_cast<void * >(&initialized),
    };
    errorHappened = false;
    thread.start(launchThread, reinterpret_cast<void * >(params));
    while (!initialized) { Thread::yield(); }
  }

  int MPIWorker::getPauseTime() const
  {
    return pause;
  }
  int MPIWorker::getMPIRank() const
  {
    return id;
  }
  int MPIWorker::getMPISize() const
  {
    return size;
  }
  int MPIWorker::getMPIRankByTarget(const Target target) const
  {
    return mpiRanks[target];
  }
  int MPIWorker::getLocalSize() const
  {
    return cpus + gpus * slots;
  }
  int MPIWorker::getGlobalSize() const
  {
    return (int)mpiRanks.size();
  }
  int MPIWorker::getTargetForCPU(const int cpuIndex) const
  {
    return cpuRanks[cpuIndex];
  }
  int MPIWorker::getTargetForGPU(const int gpuIndex, const int slot) const
  {
    return gpuRanks[gpuIndex] + slot;
  }
  int MPIWorker::getTargetForLocalCPU(const int cpuIndex) const
  {
    return rankStart + cpuIndex;
  }
  int MPIWorker::getTargetForLocalGPU(const int gpuIndex, const int slot) const
  {
    return rankStart + cpus + gpuIndex * slots + slot;
  }

  const std::vector<int> & MPIWorker::getMPIRanks() const
  {
    return mpiRanks;
  }
  const std::vector<int> & MPIWorker::getCPURanks() const
  {
    return cpuRanks;
  }
  const std::vector<int> & MPIWorker::getGPURanks() const
  {
    return gpuRanks;
  }

  double MPIWorker::wallTime() const
  {
    return MPI_Wtime();
  }

  void MPIWorker::send(const void * const buffer, const int numBytes, const Target src, const Target dst, CommStatus * const stat, AsyncRequest * const req, const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar)
  {
    if (numBytes < 0)
    {
      fprintf(stderr, "Invalid send, number of bytes must be greater than or equal to zero.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else if (!buffer && numBytes > 0)
    {
      fprintf(stderr, "Invalid send, NULL buffer with a byte count greater than zero.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else if (dst < 0 || dst >= getGlobalSize())
    {
      fprintf(stderr, "Invalid send, destination is not valid.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    addRequest(new SendRequest(globalID, dst, buffer, numBytes, stat, req, finished, error, mutex, condVar));
  }
  void MPIWorker::recv(      void * const buffer, const int numBytes, const Target src, const Target dst, CommStatus * const stat, AsyncRequest * const req, const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar)
  {
    if (numBytes < 0)
    {
      fprintf(stderr, "Invalid receive, number of bytes must be greater than or equal to zero.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else if (!buffer && numBytes > 0)
    {
      fprintf(stderr, "Invalid receive, NULL buffer with a byte count greater than zero.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else if (src != ANY_SOURCE && (src < 0 || src >= getGlobalSize()))
    {
      fprintf(stderr, "Invalid receive, source is not valid.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else if (dst < 0 || dst >= getGlobalSize())
    {
      fprintf(stderr, "Invalid send, destination is not valid.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    addRequest(new RecvRequest(globalID, src, buffer, numBytes, stat, req, finished, error, mutex, condVar));
  }
  void MPIWorker::sendRecvReplace(void * const buffer, const int numBytes, const Target src, const Target dst, CommStatus * const stat, AsyncRequest * const req, const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar)
  {
    if (numBytes < 0)
    {
      fprintf(stderr, "Invalid sendrecv, number of bytes must be greater than or equal to zero.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else if (!buffer && numBytes > 0)
    {
      fprintf(stderr, "Invalid sendrecv, NULL buffer with a byte count greater than zero.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else if (src != ANY_SOURCE && (src < 0 || src >= getGlobalSize()))
    {
      fprintf(stderr, "Invalid sendrecv, source is not valid.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else if (dst < 0 || dst >= getGlobalSize())
    {
      fprintf(stderr, "Invalid sendrecv, destination is not valid.\n");
      fflush(stderr);
      MPI_Abort(MPI_COMM_WORLD, 0);
    }
    void * sendBuf = new char[numBytes];
    memcpy(sendBuf, buffer, numBytes);
    addRequest(new SendRecvRequest(true, globalID, dst, sendBuf, numBytes, src, buffer, numBytes, stat, req, finished, error, mutex, condVar));
  }
  void MPIWorker::barrier(const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar)
  {
    addRequest(new BarrierRequest(globalID, finished, error, mutex, condVar));
  }
  void MPIWorker::broadcast(const Target root, void * const buf, const int numBytes, const int localID, const int globalID, bool * const finished, ErrorCode * const error, Mutex * const mutex, ConditionVariable * const condVar)
  {
    addRequest(new BroadcastRequest(globalID, root, buf, numBytes, finished, error, mutex, condVar));
  }

  void MPIWorker::shutdown()
  {
    addRequest(new ShutdownRequest());
  }
  void MPIWorker::abort(const ErrorCode errorCode)
  {
    errorHappened = true;
    shutdown();
  }

  bool MPIWorker::isIdle()
  {
    bool ret;
    qMutex.lock();
    ret = pendingReqs.empty() && inFlightReqs.empty();
    qMutex.unlock();
    return ret;
  }
  void MPIWorker::waitForShutdown()
  {
    thread.waitFor();
  }

  int MPIWorker::getGlobalCPUCount() const
  {
    return static_cast<int>(cpuRanks.size());
  }
  int MPIWorker::getGlobalGPUCount() const
  {
    return static_cast<int>(gpuRanks.size());
  }
}
