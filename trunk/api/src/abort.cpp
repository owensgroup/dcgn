#include <dcgn/dcgn.h>
#include <dcgn/CPUWorker.h>
#include <dcgn/GPUWorker.h>
#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>

namespace dcgn
{
  void abort(const ErrorCode errorCode)
  {
    profiler->add("Error encountered - '%s' Exiting thread.", dcgn::getErrorString(errorCode));
    for (int i = 0; i < (int)gpuWorkers.size(); ++i) gpuWorkers[i]->abort(errorCode);
    mpiWorker->abort(errorCode);
    for (int i = 0; i < (int)cpuWorkers.size(); ++i) cpuWorkers[i]->abort(errorCode);
  }
}
