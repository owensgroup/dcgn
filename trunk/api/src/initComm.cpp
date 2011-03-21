#include <dcgn/dcgn.h>

#include <dcgn/MPIWorker.h>
#include <dcgn/Profiler.h>
#include <dcgn/Request.h>

#include <cstdlib>
#include <cstdio>

namespace dcgn
{
  void initComm(const int pollPauseMS)
  {
    if (mpiWorker == 0)
    {
      fprintf(stderr, "Error, dcgn::initComm cannot be called before dcgn::init.\n");
      fflush(stderr);
      exit(1);
    }
    mpiWorker->setPauseTime(pollPauseMS);
  }
}
