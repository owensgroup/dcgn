#ifndef __DCGN_SHUTDOWNREQUEST_H__
#define __DCGN_SHUTDOWNREQUEST_H__

#include <dcgn/IORequest.h>

namespace dcgn
{
  class ShutdownRequest : public IORequest
  {
    protected:
      ErrorCode errorCode;
    public:
      ShutdownRequest();
      virtual ~ShutdownRequest();

      virtual void profileStart();
      virtual bool poll(std::vector<IORequest * > & ioRequests);
      virtual std::string toString() const;
  };
}

#endif
