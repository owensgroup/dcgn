#ifndef __DCGN_PROFILER_H__
#define __DCGN_PROFILER_H__

#include <dcgn/Mutex.h>
#include <dcgn/ThreadSpecificData.h>
#include <vector>
#include <string>

namespace dcgn
{
  class Profiler
  {
    protected:
      struct IOTime
      {
        double startTime;
        double firstPollTime;
        double commStartTime;
        double commEndTime;
        double localStartTime;
        double localEndTime;
        double signalTime;
        double destTime;
      };
      Mutex mutex;
      bool enabled;
      ThreadSpecificData tid;
      std::vector<std::vector<std::string> > events;
      std::vector<std::vector<IOTime> > ioTimes;
      double initTime;
      std::string logFileName;

      double getTime() const;
      unsigned int ensureThreadAdded();
      void addEvent(const int index, const std::string & s);
    public:
      Profiler();
      ~Profiler();

      void setEnabled(const bool b);
      bool isEnabled() const;

      void setTitle(const char * const s, ...);
      void add(const char * s, ...);
      std::string getAllEvents() const;
      void addTime(const int requestType,
                   const double startTime,    const double firstPollTime,   const double commStartTime,
                   const double commEndTime,  const double localStartTime,  const double localEndTime,
                   const double signalTime,   const double destTime);
      std::string getTimes() const;
  };
  extern Profiler * profiler;
}

#endif
