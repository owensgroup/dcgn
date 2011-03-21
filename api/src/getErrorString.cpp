#include <dcgn/dcgn.h>

namespace dcgn
{
  static const char * const ERROR_STRINGS[] =
  {
    "No error.",
    "MPI error.",
    "GPU error.",
    "CPU error.",
    "MPI thread timeout.",
    "GPU thread timeout.",
    "Job aborted.",
    "Unknown error.",
    "Invalid error code, no error string registered."
  };

  const char * getErrorString(const ErrorCode code)
  {
    if (code < 0 || code >= DCGN_NUM_ERRORS)
    {
      return ERROR_STRINGS[DCGN_NUM_ERRORS];
    }
    return ERROR_STRINGS[code];
  }
}
