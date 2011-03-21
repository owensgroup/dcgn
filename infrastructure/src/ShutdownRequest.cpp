#include <dcgn/ShutdownRequest.h>
#include <dcgn/Profiler.h>
#include <dcgn/dcgn.h>

namespace dcgn
{
  ShutdownRequest::ShutdownRequest()
    : IORequest(REQUEST_TYPE_SHUTDOWN, 0, &errorCode, 0, 0)
  {
  }
  ShutdownRequest::~ShutdownRequest()
  {
  }

  void ShutdownRequest::profileStart()
  {
    profiler->add("Servicing shutdown request.");
  }
  bool ShutdownRequest::poll(std::vector<IORequest * > & ioRequests)
  {
    return true;
  }
  std::string ShutdownRequest::toString() const
  {
    return "ShutdownRequest()";
  }
}
