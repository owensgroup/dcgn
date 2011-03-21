#ifndef __DCGN_REQUEST_H__
#define __DCGN_REQUEST_H__

#include <dcgn/dcgn.h>

namespace dcgn
{
  extern const char * REQUEST_STRINGS[REQUEST_TYPE_NUM_REQUESTS];

  /**
   * @desc Holds all the pertinent information for requests. A generic request is held in the variable <em>request</em>
   *       and the type of request is specified by <em>RequestType</em>. <em>localOrigin</em> is the local thread ID of the
   *       of the request. FYI: All slots from a single GPU-controlling thread share the same thread ID.
   */
  typedef struct _Request
  {
    RequestType type;
    void * request;
    Target localOrigin;
    bool finished;
  } Request;
  /**
   * @desc A send request from one thread to another.
   */
  /**
   * @desc A request to run a CPU kernel. <em>func</em> is the kernel, <em>param</em> is the parameter to
   *       be passed to the kernel.
   */
  typedef struct _CPUKernelRequest
  {
    CPUKernelFunction func;
    void * param;
  } CPUKernelRequest;
  /**
   * @desc A request to run a GPU kernel. <em>func</em> is the __global__ wrapper for the kernel,
   *       <em>param</em> is the parameter to be passed to the kernel. Since the grid size and block
   *       size might be dependent on other data, they can be passed in via this struct to the wrapper
   *       function.
   */
  typedef struct _GPUKernelRequest
  {
    GPUKernelFunction func;
    GPUCleanupFunction dtor;
    void * param;
    uint3 gridSize;
    uint3 blockSize;
    int sharedMemSize;
    cudaStream_t stream;
  } GPUKernelRequest;
  /**
   * @desc Initialization data needed by each CPU kernel thread. <em>initialized</em> and <em>error</em> are
   *       out-going parameters.
   */
  typedef struct _CPUInitData
  {
    int localID;
    int globalID;
    bool initialized;
    bool error;
  } CPUInitData;
  /**
   * @desc Initialization data needed by each GPU kernel thread. <em>initialized</em> and <em>error</em> are
   *       out-going parameters.
   */
  typedef struct _GPUInitData
  {
    int deviceID;
    int gpuThreadID;
    int numSlots;
    int sleepMS;
    bool initialized;
    bool error;
  } GPUInitData;
  /**
   * @desc Initialization data needed by the MPI thread. <em>initialized</em> and <em>error</em> are
   *       out-going parameters.
   */
  typedef struct _MPIInitData
  {
    int * argc;
    char *** argv;
    int totalCPUs;
    int totalGPUSlots;
    bool initialized;
    bool error;
  } MPIInitData;
}

#endif
