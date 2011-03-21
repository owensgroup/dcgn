#include <dcgn/Profiler.h>
#include <dcgn/MPIWorker.h>
#include <cstdarg>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>

namespace dcgn
{
  Profiler * profiler;
  double Profiler::getTime() const
  {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1000000.0;
  }
  unsigned int Profiler::ensureThreadAdded()
  {
    void * t = tid.getValue();
    if (!t)
    {
      mutex.lock();
      std::vector<std::string> newvec;
      unsigned int * ptr = new unsigned int;
      *ptr = events.size();
      t = reinterpret_cast<void * >(ptr);
      tid.setValue(t);
      events.push_back(newvec);
      mutex.unlock();
    }
    return *reinterpret_cast<unsigned int * >(t);
  }
  void Profiler::addEvent(const int index, const std::string & s)
  {
#if 1
    if (mpiWorker)
    {
      if (events[index].empty())
      {
        fprintf(stderr, "%s started\n", s.c_str());
      }
      else
      {
        fprintf(stderr, "%2d - %-20s: %s\n", mpiWorker->getMPIRank(), events[index].front().c_str(), s.c_str());
      }
      fflush(stderr);
    }
#endif
    FILE * fp = fopen(logFileName.c_str(), "a");
    fprintf(fp, "%2d: %s\n", index, s.c_str());
    fclose(fp);

    events[index].push_back(s);
  }

  Profiler::Profiler()
  {
    char buf[90];
    initTime = getTime();
    uid_t pid =getpid();
    sprintf(buf, "/tmp/profile_%d.log", static_cast<int>(pid));
    logFileName = buf;
  }
  Profiler::~Profiler()
  {
  }

  void Profiler::setEnabled(const bool b)
  {
    enabled = b;
  }
  bool Profiler::isEnabled() const
  {
    return enabled;
  }

  void Profiler::setTitle(const char * const s, ...)
  {
    static char * buf = new char[1048576];

    if (!enabled) return;
    va_list l;
    va_start(l, s);
    vsprintf(buf, s, l);
    va_end(l);

    unsigned int index = ensureThreadAdded();
    addEvent(index, buf);
  }
  void Profiler::add(const char * s, ...)
  {
    if (!enabled) return;
    char buf[2048];
    char timeBuf[40];
    va_list l;
    va_start(l, s);
    vsprintf(buf, s, l);
    va_end(l);

    unsigned int index = ensureThreadAdded();
    sprintf(timeBuf, "%20.9fs : ", getTime() - initTime);
    addEvent(index, std::string(timeBuf) + buf);
  }
  std::string Profiler::getAllEvents() const
  {
    if (!enabled) return "";
    std::string ret = "";
    for (int i = 0; i < (int)events.size(); ++i)
    {
      const std::vector<std::string> & vec = events[i];
      for (int j = 0; j < (int)vec.size(); ++j)
      {
        if (i == 0 && j == 0)
        {
          char buf[11];
          sprintf(buf, "%d", mpiWorker->getMPIRank());
          ret = "Node " + std::string(buf) + " " + vec[j] + "\n";
        }
        else
        {
          ret += vec[j] + "\n";
        }
      }
      ret += "\n";
    }
    return ret;
  }
  void Profiler::addTime(const int requestType,
                         const double startTime,    const double firstPollTime,   const double commStartTime,
                         const double commEndTime,  const double localStartTime,  const double localEndTime,
                         const double signalTime,   const double destTime)
  {
    if (!enabled) return;
    if ((int)ioTimes.size() <= requestType) ioTimes.resize(requestType + 1);
    IOTime time;
    time.startTime      = startTime;
    time.firstPollTime  = firstPollTime;
    time.commStartTime  = commStartTime;
    time.commEndTime    = commEndTime;
    time.localStartTime = localStartTime;
    time.localEndTime   = localEndTime;
    time.signalTime     = signalTime;
    time.destTime       = destTime;
    ioTimes[requestType].push_back(time);
  }
  std::string Profiler::getTimes() const
  {
    if (!enabled) return "";
    std::string ret = "";
    char buf[1024];
    for (int i = 0; i < (int)ioTimes.size(); ++i)
    {
      sprintf(buf, "%s\n", REQUEST_STRINGS[i]);
      ret += buf;
      for (int j = 0; j < (int)ioTimes[i].size(); ++j)
      {
        const IOTime & time = ioTimes[i][j];
        sprintf(buf,  "  full time: %14.10f\n"
                      "    time to first poll:       %14.10f\n"
                      "    time from poll to comm:   %14.10f\n"
                      "    time for global comm:     %14.10f\n"
                      "    time for local comm:      %14.10f\n"
                      "    time from comm to signal: %14.10f\n"
                      "    time for signal:          %14.10f\n",
                      time.destTime - time.startTime,
                      time.firstPollTime - time.startTime,
                      time.commStartTime - time.firstPollTime,
                      time.commEndTime - time.commStartTime,
                      time.localEndTime - time.localStartTime,
                      time.signalTime - time.localEndTime,
                      time.destTime - time.signalTime);
        ret += buf;
      }
      ret += "\n";
    }
    return ret;
  }
}
