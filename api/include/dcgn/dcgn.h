/** @file api/include/dcgn/dcgn.h
 * @brief dcgn/dcgn.h contains all of the function prototypes needed by application developers to use DCGN.
 * @author Jeffery A. Stuart
 * @version 0.10b
 * @date 2008 April 07
 *
 *
 */

#ifndef __DCGN_DCGN_H__
#define __DCGN_DCGN_H__

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

/**
  @mainpage
  @section getting-started Getting Started

  Below are some tips to get DCGN running quickly. They are by no means a full
  instruction set, but merely enough advice for a developer already familiar
  with MPI and CUDA to use DCGN.

  <h5>Quick Note: Threads running on the GPU are here after
  referred to as celts (short for compute elements). This is to easily distinguish
  between a CPU thread that controls a GPU, and a thread running on the GPU.
  </h5>

  <h2>Concepts in DCGN</h2>

  <h3>GPUs as autonomous compute units/first-class computation citizens</h3>

  DCGN allows GPUs to work pseudo autonomously of CPUs. To the application
  developer, the CPU is only responsible for initializing the GPU, beyond that,
  the GPU handles everything else on its own. This is exemplified by two points,
  the GPU running kernels and the GPU passing and receiving messages. Once a GPU
  kernel is started, it can run until program completion. When a GPU sends or
  receives a message, no explicit CPU intervention is required from the application
  developer.

  <h3>Virtualizing the GPU as a communication device</h3>

  While developing DCGN, we stumbled across the problem that there is no one-size-fits-all
  approach to mapping communication targets (equivalent to MPI ranks) to celts. To avoid
  forcing one paradigm on all users, DCGN allows users to specify the number of communication
  targets they would like per GPU. Each communication target on a GPU is referred to as a
  slot. Slots are explicitly referenced in every communication request. Slots are not
  celt-safe. While there is no restriction on which celts can use which slots, simultaneous
  access to a single slot is not allowed. This is due to the fact that locking mechanisms were
  left out in favor of increased performance.

  <h3>Simplified message passing between CPUs and/or GPUs</h3>

  CPUs and GPUs may now communicate with each other via a message passing
  protocol. This protocol is not MPI, though it bears similarities. One of the
  most important aspects of this message passing is the obfuscation of
  communication targets. A CPU sending to a CPU looks exactly like a CPU sending
  to a GPU. The converse is also true.

  <h2>Building DCGN</h2>

  To use DCGN, one first needs to obtain the source code and build the library.
  The Makefile variables <b>CUDA_INC_DIR</b>, <b>CUDA_LIB_DIR</b>,
  <b>MPI_INC_DIR</b>, and <b>MPI_LIB_DIR</b> need to be adjusted accordingly.
  Also, it should be stated that DCGN has only been tested on Linux systems.
  Once the Makefile variables have been modified, one can proceed to build the
  library and test examples by executing <code>make</code>.

  <h2>Including Necessary Header Files</h2>

  Once the DCGN libraries are built, one can proceed to create their first DCGN
  application. All CPU and GPU files using DCGN must include
  <code>&lt;dcgn/dcgn.h&gt;</code>. GPU (.cu) code must also include
  <code>&lt;dcgn/CUDAFunctions.h&gt;.</code>

  <h2>Initializing DCGN</h2>

  In order to use DCGN, one's CPU code must call either
  <code>dcgn::initAll</code> OR <code>dcgn::init</code>, followed by (in any
  order) <code>dcgn::initComm</code>, <code>dcgn::initCPU</code>, and
  <code>dcgn::initGPU</code>, and finally <code>dcgn::start</code>. At the end
  of one's program, one needs to call <code>dcgn::finalize</code> to ensure that
  all MPI and GPU resources are reclaimed properly.

  One should notice that the <code>dcgn::initGPU</code> and
  <code>dcgn::initAll</code> functions take a non-intuitive set of parameters.
  The prototype for <code>dcgn::initGPU</code> is shown below.

  <code>void initGPU(const int * const allocatedGPUs, const int slotsPerGPU, const int asyncTransfersPerSlot);</code>

  <ul>
    <li>
      <code>allocatedGPUs</code> is an array of integers, terminated by a <code>-1</code>.
      Each integer in the array is an index, such as would be passed to
      <code>cuDeviceGet</code>, corresponding to a CUDA-capable GPU. Repeated
      integers will most likely cause a runtime error or unsuccesful launch of DCGN.
    </li>
    <li>
      <code>slotsPerGPU</code> is the number of <em>slots</em> one wishes to use per GPU.
      Currently, the number of slots must be the same per GPU <i>on a per node basis</i>, not
      across the entire network of GPUs. This will be changed in future releases of DCGN to
      allow more flexibility.
    </li>
    <li>
      <code>asyncTransferPerSlot</code> is the maximum number of asynchronous transfers each
      slot supports. As the GPU must manipulate GPU global RAM and keep persistent copies of
      each unfinished asynchronous request, there is a limit on the number of communications
      that may occur simultaneously.
    </li>
  </ul>

  One runs their code via kernels, both on the CPU and on the GPU. Kernel-launch
  requests are made via calls to DCGN. Once the specified CPU thread or GPU
  becomes available, the kernel is scheduled and ran.

  To communicate via point-to-point, one initiates a send (either synchronous or
  asynchronous) and a corresponding receive. The send can happen from a CPU or
  GPU, and it can be to a CPU or GPU. There are no restrictions on the end
  points of a communication.

  When one uses a point-to-point communication on the GPU, the memory used must
  reside in GPU global RAM. Registers and shared memory are not compatible with
  DCGN. However, other parameters, such as a <code>dcgn::CommStatus</code> can
  reside in registers or shared memory.

  To use collective communications, one needs to ensure that EVERY communication
  target executes the collective communication. This means that if one is using
  two GPUs, each with five slots, all ten slots must participate in the
  collective, as well as every CPU-kernel thread.

*/

/**
 * All publicly exposed functions of DCGN reside in the dcgn namespace.
 */
namespace dcgn
{
  /**
   * A target is a source or sink for communication. This is equivalent to a rank in MPI.
   */
  typedef long long Target;
  /**
   * This is no longer used.
   */
  typedef long long KernelID;

  /**
   * An identifier denoting that a communication may be sourced from any valid target. It is often the
   * case that some receives can be sourced from any valid communication target. Instead of initializing
   * many receives or performing a probe function, one may use this identifier to denote that a receive
   * may be sourced from any valid communication target.
   */
  const Target ANY_SOURCE = -1;

  /**
   * DCGN functions return error codes. One may examine the actual error code to determine the cause of
   * failure of a function. These error codes are also used internally.
   */
  typedef enum _ErrorCode
  {
    /** A return value DCGN_ERROR_NONE indicates that no error occurred. */
    DCGN_ERROR_NONE = 0,
    /** A return value DCGN_ERROR_MPI indicates that the underlying MPI library returned an error. */
    DCGN_ERROR_MPI,
    /** A return value DCGN_ERROR_GPU indicates that the GPU (CUDA) encountered an error. */
    DCGN_ERROR_GPU,
    /** A return value DCGN_ERROR_CPU indicates that a CPU thread signalled an error. */
    DCGN_ERROR_CPU,
    /** A return value DCGN_ERROR_MPI_TIMEOUT indicates that an MPI thread timed out and requested program termination. */
    DCGN_ERROR_MPI_TIMEOUT,
    /** A return value DCGN_ERROR_GPU_TIMEOUT indicates that a GPU thread timed out and requested program termination. */
    DCGN_ERROR_GPU_TIMEOUT,
    /** A return value DCGN_ERROR_ABORTED indicates that dcgn::abort was called prior to the current function's execution finishing. */
    DCGN_ERROR_ABORTED,
    /** A return value DCGN_ERROR_UNKNOWN indicates that an unknown error occurred. */
    DCGN_ERROR_UNKNOWN,

    /** THis value is used to indicate the number of distinct error values a dcgn function is capable of returning. */
    DCGN_NUM_ERRORS,
  } ErrorCode;

  /**
   * A CommStatus object is often used by DCGN to relay parameters of a communication request back to the caller.
   */
  typedef struct _CommStatus
  {
    /** The source of the communication. */
    Target src;
    /** The destination target of the communication. */
    Target dst;
    /** The error code returned by DCGN from executing the communication. */
    ErrorCode errorCode;
    /** The number of data bytes transmitted via the communication. */
    int numBytes;
  } CommStatus;

  /**
   * An AsyncRequest object is used to allow users to query whether an asynchronous DCGN communication completed.
   */
  typedef struct _AsyncRequest
  {
    /** One can check the value of the completed flag to determine if an asynchronous communication finished. */
    bool completed;
    /** Upon a completion of a communication request, the stat field is updated to reflect the properties of the communication. */
    CommStatus stat;
  } AsyncRequest;

  /**
   * Enumerates the possible request types sent between threads. Application
   * developers will never need to use any members of this enumeration.
   */
  typedef enum _RequestType
  {
    /**
     * No request was actually specified.
     */
    REQUEST_TYPE_NONE = 0,

    /**
     * Send data from one thread to another (possibly non-local) thread.
     */
    REQUEST_TYPE_SEND,
    /**
     * Receive data sent from another (possibly non-local) thread.
     */
    REQUEST_TYPE_RECV,
    /**
     * Send and recv data between threads with one call.
     */
    REQUEST_TYPE_SEND_RECV,
    /**
     * Send and recv data between threads with one call.
     */
    REQUEST_TYPE_SEND_RECV_REPLACE,
    /**
     * Barrier with all the threads.
     */
    REQUEST_TYPE_BARRIER,
    /**
     * Broadcast info from one thread to all other threads.
     */
    REQUEST_TYPE_BROADCAST,


    /**
     * Run a kernel on a CPU.
     */
    REQUEST_TYPE_CPU_KERNEL,
    /**
     * Run a kernel on a GPU.
     */
    REQUEST_TYPE_GPU_SETUP,
    /**
     * Setup state and dynamic memory for a GPU kernel (must be done within the GPU thread).
     */
    REQUEST_TYPE_GPU_KERNEL,

    /**
     * Everything is done so shutdown.
     */
    REQUEST_TYPE_SHUTDOWN,

    /**
     * Allocate global GPU memory. GPU malloc! Why didn't I ever think of this before?
     */
    REQUEST_TYPE_MALLOC,
    /**
     * Free global GPU memory. GPU free! I thought of this right after malloc...
     */
    REQUEST_TYPE_FREE,

    REQUEST_TYPE_NUM_REQUESTS,
  } RequestType;


  class OutputStream;

  /**
   * A data structure internal to DCGN, application developers will never need to use this structure. As I/O requests are
   * made by a GPU, a structure of this type is filled out in global GPU memory. The CPU will then copy this structure from
   * the GPU to the CPU, examine its contents, and execute the requested communication.
   */
  typedef struct _GPUIORequest
  {
    /**
     * The type of communication request
     */
    RequestType type;
    /**
     * The size of the communication.
     */
    int numBytes;
    /**
     * The buffer used for communication.
     */
    void * buf;
    /**
     * The source of the communication.
     */
    Target from;
    /**
     * The destination of the communication.
     */
    Target to;
    /**
     * A flag pollable by the GPU that indicates if the requested
     * communication is completed. An integer is used for memory
     * alignment purposes.
    */
    int done;
    /**
     * A private variable used by the GPU to execute a spin loop.
     */
    AsyncRequest req;
  } GPUIORequest;
  /**
   * This data structure is used to communication library parameters to the GPU initialization code.
   */
  typedef struct _GPUInitRequest
  {
    /**
     * A pointer to GPU global memory where each slot's GPUIORequest lies.
     */
    GPUIORequest * gpuReqs;
    /**
     * A pointer to GPU global memory where each slot's asynchronous GPUIORequest lies.
     */
    GPUIORequest * gpuAsyncReqs;
    /**
     * The global node ID of the node on which this GPU lies. Used internally to determine communication parameters.
     */
    int gpuNodeID;
    /**
     * The maximum number of asynchronous requests per slot that can be pending.
     */
    int numAsyncReqs;
    /**
     * The base rank of this GPU. This is equivalent to the rank of the 0<sup>th</sup> slot on the GPU.
     */
    Target gpuRank;
    /**
     * The number of slots on this GPU.
     */
    Target gpuSize;
    /**
     * Unused for now. Once full string support is supported by the GPU, this will hold an OutputStream stream object usable by
     * application developers
     */
    OutputStream * outputStream;
    /**
     * The global ranks of all CPUs.
     */
    int * cpuRanks;
    /**
     * The global ranks of all GPUs. This is actually all the ranks of the 0<sup>th</sup> slots on all GPUs.
     */
    int * gpuRanks;
    /**
     * An array of asynchronous request handles for use by DCGN on the GPU.
     */
    AsyncRequest ** localAsyncReqs;
    /**
     * Debugging information holder. One should ignore this.
     */
    unsigned long long * debugInfo;
  } GPUInitRequest;

  /**
   * The prototype for CPU-kernel routines. This is similar to POSIX-thread start functions except
   * in the fact that a kernel function does not return a value.
   *
   * @param param A user specified parameter. Made a void * for full flexibility.
   */
  typedef void (* CPUKernelFunction)(void * param);
  /**
   * The prototype for GPU-kernel cleanup routines.
   *
   * @param param A user specified parameter. Made a void * for full flexibility.
   */
  typedef void (* GPUCleanupFunction)(void * param);
  /**
   * The prototype for GPU-kernel functions. GPU-kernel functions are actually CPU wrappers that call down to __global__ device functions
   * by using either the driver API or the runtime API.
   *
   * @param kernelParam   A user specified parameter. Made a void * for full flexibility.
   *
   * @param libParam      The configuration parameters for DCGN. This variable must be passed down to the GPU and used in a call to
   *                      dcgn::gpu::init to ensure that the library initializes properly.
   *
   * @param gridSize      The user requested grid size. This may be ignored in favor of a different grid size.
   *
   * @param blockSize     The user requested block size. This may be ignored in favor of a different block size.
   *
   * @param sharedMemSize The user requested shared memory size. This may be ignored in favor of a different shared memory size. If a user
   *                      chooses to specify the size of shared memory they request, they must take into account that DCGN uses a small
   *                      amount of shared memory.
   *
   * @param stream        The stream used by DCGN to monitor the kernel's progress; This variable <i><b>MUST</b></i> be passed as part of the
   *                      execution configuration for the GPU kernel.
   */
  typedef void (* GPUKernelFunction)(void * kernelParam, const GPUInitRequest libParam, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize, cudaStream_t * const stream);

  /**
   * A single initialization function for the DCGN library that combines the calls five other functions. This function must be executed on the
   * CPU before any other DCGN function. If this is not done, at best an unknown value will be returned. At worst, program failure will occur.
   * This function currently initializes all threads on the CPU, initializes MPI, and ensures that DCGN is in a ready state to begin execution
   * of GPU kernels, GPU kernels, and communication requests.
   *
   * @param argc              A pointer to an integer representing the number of command line arguments passed to the program.
   * @param argv              A pointer to the array of command line arguments for the program.
   * @param allocatedCPUs     The number of CPU cores allocated for this program on this local node.
   * @param allocatedGPUs     The number and indices of all GPUs to be used on this node. This array MUST be <code>-1</code> terminated.
   * @param numSlotsPerGPU    The number of slots per GPU requested on this node.
   * @param numAsyncGPUTrans  The maximum number of asynchronous communications per slot requested for each GPU.
   * @param pollPauseMS       The amount of time to sleep after each poll of GPU memory or round of <i>MPI_Test</i>s.
   *                          <ul>
   *                            <li> pollPauseMS <  0 - Don't sleep, just poll/test again.</li>
   *                            <li> pollPauseMS == 0 - Yield.<li>
   *                            <li> pollPauseMS >  0 - Sleep for specified number of milliseconds.</li>
   *                          </ul>
   *
   * @sa init, initComm, initCPU, initGPU, start
   */
  void initAll(int * argc, char *** argv, const int allocatedCPUs, const int * const allocatedGPUs, const int numSlotsPerGPU, const int numAsyncGPUTrans, const int pollPauseMS);
  /**
   * The first in a series of initialization function that need to be called for the DCGN library. This function actually creates and enables
   * the profiler and creates the MPI thread.
   *
   * @param argc              A pointer to an integer representing the number of command line arguments passed to the program.
   * @param argv              A pointer to the array of command line arguments for the program.
   *
   * @sa initAll, initComm, initCPU, initGPU, start
   */
  void init(int * argc, char *** argv);
  /**
   * Used to enable the MPI thread. This function cannot be called before init.
   *
   * @param pollPauseMS       The amount of time to sleep after each poll of GPU memory or round of <i>MPI_Test</i>s.
   *                          <ul>
   *                            <li> pollPauseMS <  0 - Don't sleep, just poll/test again.</li>
   *                            <li> pollPauseMS == 0 - Yield.<li>
   *                            <li> pollPauseMS >  0 - Sleep for specified number of milliseconds.</li>
   *                          </ul>
   *
   * @sa initAll, init, initCPU, initGPU, start
   */
  void initComm(const int pollPauseMS);
  /**
   * Used to create and enable the CPU threads. This function cannot be called before init.
   *
   * @param allocatedCPUs     The number of CPU cores allocated for this program on this local node.
   *
   * @sa initAll, init, initComm, initGPU, start
   */
  void initCPU(const int allocatedCPUs);
  /**
   * Used to create and enable the GPU threads. This function cannot be called before init.
   *
   * @param allocatedGPUs           The number of CPU cores allocated for this program on this local node.
   * @param slotsPerGPU             The number of slots per GPU requested on this node.
   * @param asyncTransfersPerSlot   The maximum number of asynchronous communications per slot requested for each GPU.
   *
   * @sa initAll, init, initComm, initCPU, start
   */
  void initGPU(const int * const allocatedGPUs, const int slotsPerGPU, const int asyncTransfersPerSlot);
  /**
   * To be called after init, initComm, initCPU, and initGPU. Once this function is called, DCGN is considered to be
   * completely initialized and ready to go.
   *
   * @sa initAll, init, initComm, initCPU, initGPU
   */
  void start();
  /**
   * Called to shut down DCGN. Any kernel launch requests made prior to this call will be serviced, as well as any communication
   * requests necessary to complete the queued kernels.
   */
  void finalize();
  /**
   * Called to cause an immediate end to the DCGN program. Due to problems with CUDA, any GPU kernels must finish execution before
   * an abort takes complete effect. This can cause deadlock, and thus this function is highly dangerous and it is recommended that its
   * use be avoided.
   *
   * @param errorCode   The user specified error code to report.
   */
  void abort(const ErrorCode errorCode);

  /**
   * Used to send from a CPU thread to another communication target. This function can only be called from CPU-kernel
   * threads created by DCGN.
   *
   * @param dst         The global ID for the destination of this communication.
   * @param buffer      The data to be sent.
   * @param numBytes    The number of bytes to be sent.
   *
   * @sa dcgn::gpu::send
   */
  void send(const Target dst, const void * const buffer, const int numBytes);
  /**
   * Receive via a CPU thread from another communication target. This function can only be called from CPU-kernel
   * threads created by DCGN.
   *
   * @param src         The global ID for the source of this communication.
   * @param buffer      The data to be sent.
   * @param numBytes    The maximum number of bytes to be received.
   * @param stat        An outward variable used to convey the number of bytes actually received.
   *
   * @sa dcgn::gpu::recv
   */
  void recv(const Target src,       void * const buffer, const int numBytes, CommStatus * const stat);
  /**
   * Send and recieve from a CPU thread to another communication target. The data in the buffer is first sent, then overwritten
   * with the receive data. This function can only be called from CPU-kernel threads created by DCGN.
   *
   * @param dst         The global ID for the destination of this communication.
   * @param src         The global ID for the source of this communication.
   * @param buffer      The data to be sent and received.
   * @param numBytes    The number of bytes to be sent, and the maximum number of bytes to be received.
   * @param stat        An outward variable used to convey the number of bytes actually received.
   *
   * @sa dcgn::gpu::sendRecvReplace
   */
  void sendRecvReplace(const Target dst, const Target src, void * const buffer, const int numBytes, CommStatus * const stat);
  /**
   * Take place in a barrier across ALL global communication targets. Every global communication target, including every slot on each
   * GPU, must call their appropriate barrier function.
   *
   * @sa dcgn::gpu::barrier
   */
  void barrier();
  /**
   * Take place in a broadcast across ALL global communication targets.Every global communication target, including every slot on each
   * GPU, must call their appropriate barrier function.
   *
   * @param root        The global ID for the root of the broadcast.
   * @param bytes       The buffer used for communication.
   * @param numBytes    The number of bytes to be broadcasted. This must be the same on every node.
   *
   * @sa dcgn::gpu::broadcast
   */
  void broadcast(const Target root, void * const bytes, const int numBytes);

  /**
   * Send asynchronously from a CPU thread to another communication target. This function can only be called from CPU-kernel
   * threads created by DCGN. Control returns immediately from this function, even though the communication most likely did not complete.
   *
   * @param dst         The global ID for the destination of this communication.
   * @param buffer      The data to be sent.
   * @param numBytes    The number of bytes to be sent.
   * @param req         An object used to poll the completion status of an asyncrhonous communication.
   *
   * @sa dcgn::gpu::asyncSend
   */
  void asyncSend(const Target dst, const void * const buffer, const int numBytes, AsyncRequest * const req);
  /**
   * Receive asynchronously via a CPU thread from another communication target. This function can only be called from CPU-kernel
   * threads created by DCGN. Control returns immediately from this function, even though the communication most likely did not complete.
   *
   * @param src         The global ID for the source of this communication.
   * @param buffer      The data to be sent.
   * @param numBytes    The maximum number of bytes to be received.
   * @param req         An object used to poll the completion status of an asyncrhonous communication.
   *
   * @sa dcgn::gpu::asyncRecv
   */
  void asyncRecv(const Target src,       void * const buffer, const int numBytes, AsyncRequest * const req);
  /**
   * Tests an asynchronous communication for completion in a non-blocking manner. Stores the results of the communication (destination,
   * number of bytes, etc) in stat for further examination.
   *
   * @param req         Handle to asynchronous communication request.
   * @param stat        Outward variable used to convey parameters about the communication.
   *
   * @return <b>true</b> if the communication completed, <b>false</b> otherwise.
   *
   * @sa asyncSend asyncRecv asyncWait
   */
  bool asyncTest(AsyncRequest * const req, CommStatus * const stat);
  /**
   * Waits for an asynchronous communication for completion in a blocking manner. Stores the results of the communication (destination,
   * number of bytes, etc) in stat for further examination.
   *
   * @param req         Handle to asynchronous communication request.
   * @param stat        Outward variable used to convey parameters about the communication.
   *
   * @sa asyncSend asyncRecv asyncTest
   */
  void asyncWait(AsyncRequest * const req, CommStatus * const stat);

  /**
   * Queues a CPU kernel for launch. Once the specified CPU thread becomes available, the kernel is executed and run to completion.
   *
   * @param cpuThreadIndex    The index of the thread on which to run this kernel.
   * @param func              The kernel function.
   * @param param             A user specified parameter.
   */
  void launchCPUKernel(const int cpuThreadIndex, const CPUKernelFunction func, void * const param);
  /**
   * Queues a GPU kernel for launch. Once the specified GPU thread becomes available, the kernel is executed and run to completion.
   *
   * @param gpuThreadIndex    The index of the thread on which to run this kernel.
   * @param func              A CPU-wrapper function that executes the kernel.
   * @param dtor              A CPU function that cleans up the GPU resources used by the kernel.
   * @param param             A user specified parameter.
   * @param gridSize          The requested size of the grid. Can be overridden in <b>func</b>.
   * @param blockSize         The requested size of each block. Can be overridden in <b>func</b>.
   * @param sharedMemSize     The requested size of shared memory.
   */
  void launchGPUKernel(const int gpuThreadIndex, const GPUKernelFunction func, const GPUCleanupFunction dtor, void * const param, const uint3 & gridSize, const uint3 & blockSize, const int sharedMemSize = 0);

  /**
   * Determines if the specified CPU thread is idle.
   *
   * @param localCPUIndex     The zero-based index of the CPU thread.
   *
   * @return <b>true</b> if the thread is idle (is not working and has no work in its queue), <b>false</b> otherwise.
   */
  bool isCPUIdle(const int localCPUIndex);
  /**
   * Determines if the specified GPU is idle.
   *
   * @param localGPUIndex     The zero-based index of the GPU. This argument does not correspond to the physical integer address of the
   *                          GPU. Given a local process with N GPUs, the indices are in the range [0,N-1].
   *
   * @return <b>true</b> if the GPU is idle (is not working and has no work in its queue), <b>false</b> otherwise.
   */
  bool isGPUIdle(const int localGPUIndex);
  /**
   * Determines if all CPU threads and GPUs are idle. This function is prone to race conditions, you probably should not use it.
   *
   * @return <b>true</b> if all resources are idle (none are working and none have work in their queue), <b>false</b> otherwise.
   *
   * @warning This function is really prone to race conditions, don't rely too heavily on the output.
   */
  bool areAllLocalResourcesIdle(); // prone to race conditions, beware

  /**
   * Checks the entire job (not just the local process) and returns how many CPU-kernel threads are used.
   *
   * @return The total number of CPU-kernel threads employed by this job.
   */
  int globalCPUCount();
  /**
   * Checks the entire job (not just the local process) and returns how many GPUs are used.
   *
   * @return The total number of GPUs employed by this job.
   */
  int globalGPUCount();

  /**
   * Returns the current time accurate to fractions of a microsecond. This is a direct call to MPI_Wtime().
   *
   * @return the current time in seconds.
   */
  double wallTime();

  /**
   * Returns the job (MPI) rank of the process. This value is the same for all CPU and GPU threads.
   *
   * @return The MPI rank of the process.
   */
  int getNodeID();

  /**
   * Returns the global rank of the thread or GPU slot.
   *
   * @return The global rank of the thread or GPU slot.
   */
  Target getRank();
  /**
   * Returns the combined number of CPU threads and GPUs employed by this job.
   *
   * @return The combined number of CPU threads and GPUs employed by this job.
   */
  Target getSize();
  /**
   * Gets the communication ID of the specified CPU thread.
   *
   * @param cpu       The global index of the CPU-kernel thread.
   *
   * @return The communication ID of the specified CPU thread.
   */
  Target getCPUID(const int cpu);
  /**
   * Gets the communication ID of the slot on the specified GPU.
   *
   * @param gpu       The global index of the GPU.
   * @param slot      The slot on the GPU.
   *
   * @return The communication ID of the specified CPU thread.
   */
  Target getGPUID(const int gpu, const int slot);

  /**
   * Returns a statically allocated string describing the specified error.
   *
   * @param code      The error code in need of description.
   *
   * return A description of the error.
   */
  const char * getErrorString(const ErrorCode code);

  /**
   * Don't use this function yet.
   *
   * @return something you shouldn't use.
   */
  OutputStream & output();

  /**
   * All publicly exposed functions of DCGN for use on the GPU reside in the dcgn::gpu namespace.
   */
  namespace gpu
  {
    /**
     * Initializes all the necessary communication variables on the GPU.
     *
     * @param initReq     The DCGN library's parameters.
     */
    __device__ void init(const GPUInitRequest initReq);

    /**
     * Used to send from a GPU slot to another communication target. This function can only be called by the GPU, and only
     * after init has been called.
     *
     * @param slot        The communication slot to use for the communication.
     * @param dst         The global ID for the destination of this communication.
     * @param buffer      The data to be sent.
     * @param numBytes    The number of bytes to be sent.
     *
     * @sa dcgn::send
     */
    __device__ void send(const int slot, const Target dst, const void * const buffer, const int numBytes);
    /**
     * Receive via a GPU from another communication target. This function can only be called by the GPU, and only
     * after init has been called.
     *
     * @param slot        The communication slot to use for the communication.
     * @param src         The global ID for the source of this communication.
     * @param buffer      The data to be sent.
     * @param numBytes    The maximum number of bytes to be received.
     * @param stat        An outward variable used to convey the number of bytes actually received.
     *
     * @sa dcgn::recv
     */
    __device__ void recv(const int slot, const Target src,       void * const buffer, const int numBytes, CommStatus * const stat);
    /**
     * Send and recieve from a GPU to another communication target. The data in the buffer is first sent, then overwritten
     * with the receive data. This function can only be called by the GPU, and only after init has been called.
     *
     * @param slot        The communication slot to use for the communication.
     * @param dst         The global ID for the destination of this communication.
     * @param src         The global ID for the source of this communication.
     * @param buffer      The data to be sent and received.
     * @param numBytes    The number of bytes to be sent, and the maximum number of bytes to be received.
     * @param stat        An outward variable used to convey the number of bytes actually received.
     *
     * @sa dcgn::sendRecvReplace
     */
    __device__ void sendRecvReplace(const int slot, const Target dst, const Target src, void * const buffer, const int numBytes, CommStatus * const stat);
    /**
     * Take place in a barrier across ALL global communication targets. Every global communication target, including every slot on each
     * GPU, must call their appropriate barrier function.
     *
     * @param slot        The communication slot to use for the barrier.
     *
     * @sa dcgn::barrier
     */
    __device__ void barrier(const int slot);
    /**
     * Take place in a broadcast across ALL global communication targets.Every global communication target, including every slot on each
     * GPU, must call their appropriate barrier function.
     *
     * @param slot        The communication slot to use for the broadcast.
     * @param root        The global ID for the root of the broadcast.
     * @param bytes       The buffer used for communication.
     * @param numBytes    The number of bytes to be broadcasted. This must be the same on every node.
     *
     * @sa dcgn::broadcast
     */
    __device__ void broadcast(const int slot, const Target root, void * const bytes, const int numBytes);

    /**
     * Send asynchronously via a GPU to another communication target. This function can only be called from the GPU. Control
     * returns immediately from this function, even though the communication most likely did not complete.
     *
     * @param slot        The communication slot to use.
     * @param dst         The global ID for the destination of this communication.
     * @param buffer      The data to be sent.
     * @param numBytes    The number of bytes to be sent.
     * @param req         An object used to poll the completion status of an asyncrhonous communication.
     *
     * @sa dcgn::gpu::asyncSend
     */
    __device__ void asyncSend(const int slot, const Target dst, const void * const buffer, const int numBytes, AsyncRequest * const req);
    /**
     * Receive asynchronously via a CPU thread from another communication target. This function can only be called from CPU-kernel
     * threads created by DCGN. Control returns immediately from this function, even though the communication most likely did not complete.
     *
     * @param slot        The communication slot to use.
     * @param src         The global ID for the source of this communication.
     * @param buffer      The data to be sent.
     * @param numBytes    The maximum number of bytes to be received.
     * @param req         An object used to poll the completion status of an asyncrhonous communication.
     *
     * @sa dcgn::gpu::asyncRecv
     */
    __device__ void asyncRecv(const int slot, const Target src,       void * const buffer, const int numBytes, AsyncRequest * const req);
    /**
     * Tests an asynchronous communication for completion in a non-blocking manner. Stores the results of the communication (destination,
     * number of bytes, etc) in stat for further examination.
     *
     * @param slot        The communication slot to use.
     * @param req         Handle to asynchronous communication request.
     * @param stat        Outward variable used to convey parameters about the communication.
     *
     * @return <b>true</b> if the communication completed, <b>false</b> otherwise.
     *
     * @sa asyncSend asyncRecv asyncWait
     */
    __device__ bool asyncTest(const int slot, AsyncRequest * const req, CommStatus * const stat);
    /**
     * Waits for an asynchronous communication for completion in a blocking manner. Stores the results of the communication (destination,
     * number of bytes, etc) in stat for further examination.
     *
     * @param slot        The communication slot to use.
     * @param req         Handle to asynchronous communication request.
     * @param stat        Outward variable used to convey parameters about the communication.
     *
     * @sa asyncSend asyncRecv asyncTest
     */
    __device__ void asyncWait(const int slot, AsyncRequest * const req, CommStatus * const stat);

    /**
     * Returns the job (MPI) rank of the process. This value is the same for all CPU and GPU threads.
     *
     * @return The MPI rank of the process.
     */
    __device__ int getNodeID();
    /**
     * Returns the global rank of GPU slot.
     *
     * @param slot        The communication slot.
     *
     * @return The global rank of the thread or GPU slot.
     */
    __device__ Target getRank(const int slot);
    /**
     * Returns the combined number of CPU threads and GPUs employed by this job.
     *
     * @return The combined number of CPU threads and GPUs employed by this job.
     */
    __device__ Target getSize();

    /**
     * Gets the communication ID of the specified CPU thread.
     *
     * @param cpu       The global index of the CPU-kernel thread.
     *
     * @return The communication ID of the specified CPU thread.
     */
    __device__ Target getCPUID(const int cpu);
    /**
     * Gets the communication ID of the slot on the specified GPU.
     *
     * @param gpu       The global index of the GPU.
     * @param slot      The slot on the GPU.
     *
     * @return The communication ID of the specified CPU thread.
     */
    __device__ Target getGPUID(const int gpu, const int slot);

    /**
     * Don't use this right now.
     *
     * @param slot      Ignore this.
     * @param size      Ignore this.
     *
     * @return          Ignore this.
     */
    __device__ void * malloc(const int slot, const size_t size);
    /**
     * Don't use this right now.
     *
     * @param slot      Ignore this.
     * @param ptr       Ignore this.
     */
    __device__ void free(const int slot, void * const ptr);
  }

}

#include <dcgn/Request.h>
#include <dcgn/OutputStream.h>

#endif
