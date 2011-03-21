#include <dcgn/dcgn.h>

namespace dcgn
{
  bool asyncTest(AsyncRequest * const req, CommStatus * const stat)
  {
    if (!req || !req->completed) return false;
    if (stat) *stat = req->stat;
    return true;
  }
}
