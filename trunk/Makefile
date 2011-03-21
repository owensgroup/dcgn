CUDA_INC_DIR=/usr/local/cuda/include
CUDA_LIB_DIR=/usr/local/cuda/lib64

MPI_INC_DIR   = /usr/mpi/gcc/mvapich2-1.5.1/include
MPI_LIB_DIR   = /usr/mpi/gcc/mvapich2-1.5.1/lib
MPI_LIBS      = -L $(MPI_LIB_DIR) -lmpich -libverbs -libumad -lpthread
CUDA_LIBS     = -L $(CUDA_LIB_DIR) -lcuda -lcudart
CUDA_MPI_LIBS = $(MPI_LIBS) $(CUDA_LIBS);
DCGN_LIBS     = -L . -ldcgn $(MPI_LIBS) $(CUDA_LIBS)

OPTFLAGS=-g
NVCC=nvcc
CXX=g++
CXXFLAGS =-I . -I $(CUDA_INC_DIR) -I $(MPI_INC_DIR) -I api/include -I infrastructure/include -c -Wall $(OPTFLAGS) -D MPICH_SKIP_MPICXX
NVCCFLAGS=-I . -I $(CUDA_INC_DIR) -I $(MPI_INC_DIR) -I api/include -I infrastructure/include  $(OPTFLAGS) -c -D MPICH_SKIP_MPICXX
NVCC_CXXFLAGS=-arch sm_13 --compiler-options -Wall -D __CUDA__
AR=ar
ARFLAGS=-cvr

LIBRARY=libdcgn.a

API_CXX_SRC=                                    \
  api/src/abort.cpp                             \
  api/src/areAllLocalResourcesIdle.cpp          \
  api/src/asyncRecv.cpp                         \
  api/src/asyncSend.cpp                         \
  api/src/asyncTest.cpp                         \
  api/src/asyncWait.cpp                         \
  api/src/barrier.cpp                           \
  api/src/broadcast.cpp                         \
  api/src/finalize.cpp                          \
  api/src/getCPUID.cpp                          \
  api/src/getErrorString.cpp                    \
  api/src/getGPUID.cpp                          \
  api/src/getNodeID.cpp                         \
  api/src/getRank.cpp                           \
  api/src/getSize.cpp                           \
  api/src/globalCPUCount.cpp                    \
  api/src/globalGPUCount.cpp                    \
  api/src/init.cpp                              \
  api/src/initAll.cpp                           \
  api/src/initComm.cpp                          \
  api/src/initCPU.cpp                           \
  api/src/initGPU.cpp                           \
  api/src/isCPUIdle.cpp                         \
  api/src/isGPUIdle.cpp                         \
  api/src/launchCPUKernel.cpp                   \
  api/src/launchGPUKernel.cpp                   \
  api/src/output.cpp                            \
  api/src/recv.cpp                              \
  api/src/send.cpp                              \
  api/src/sendRecvReplace.cpp                   \
  api/src/start.cpp                             \
  api/src/wallTime.cpp                          \
                                                \
  api/src/OutputStream.cpp                      \

API_CU_SRC=                                     \

INF_CXX_SRC=                                    \
  infrastructure/src/BarrierRequest.cpp         \
  infrastructure/src/BroadcastRequest.cpp       \
  infrastructure/src/CollectiveRequest.cpp      \
  infrastructure/src/ConditionVariable.cpp      \
  infrastructure/src/CPUWorker.cpp              \
  infrastructure/src/GPUWorker.cpp              \
  infrastructure/src/IORequest.cpp              \
  infrastructure/src/MPIWorker.cpp              \
  infrastructure/src/Mutex.cpp                  \
  infrastructure/src/Profiler.cpp               \
  infrastructure/src/RecvRequest.cpp            \
  infrastructure/src/SendRequest.cpp            \
  infrastructure/src/SendRecvRequest.cpp        \
  infrastructure/src/ShutdownRequest.cpp        \
  infrastructure/src/Thread.cpp                 \
  infrastructure/src/ThreadSpecificData.cpp     \

INF_CU_SRC=                                     \

SAMPLE_SRC=                                     \
  samples/cpu_send_recv.cpp                     \
  samples/cpu_async_send_recv.cpp               \
  samples/cpu_send_async_recv.cpp               \
  samples/cpu_gpu_barrier.cpp                   \
  samples/cpu_gpu_send_recv.cpp                 \
  samples/cpu_gpu_broadcast.cpp                 \
  samples/parallel_cpu_matmult.cpp              \
  samples/parallel_cpu_nbody.cpp                \
  samples/parallel_gpu_matmult.cpp              \
  samples/serial_cpu_mandelbrot.cpp             \
  samples/serial_cpu_matmult.cpp                \
  samples/serial_cpu_nbody.cpp                  \
  samples/serial_gpu_matmult.cpp                \
  samples/speed_test_cpu.cpp                    \
  samples/speed_test_cpu2.cpp                   \

SAMPLE_CU_SRC=                                  \
  samples/cpu_send_recv.cu                      \
  samples/cpu_gpu_barrier.cu                    \
  samples/cpu_gpu_send_recv.cu                  \
  samples/cpu_gpu_broadcast.cu                  \
  samples/gpu_async_send_async_recv.cu          \
  samples/gpu_matmult.cu                        \
  samples/parallel_gpu_mandelbrot.cu            \
  samples/parallel_gpu_mandelbrot2.cu           \
  samples/parallel_gpu_matmult.cu               \
  samples/parallel_gpu_nbody.cu                 \
  samples/parallel_gpu_nbody2.cu                \
  samples/parallel_gpu_nbody3.cu                \
  samples/serial_gpu_mandelbrot.cu              \
  samples/serial_gpu_nbody.cu                   \
  samples/speed_test_gpu.cu                     \

TEST_CPP_SRC=                                   \
  tests/dcgn_cpu_barrier_0.cpp                  \
  tests/dcgn_cpu_barrier_1.cpp                  \
  tests/dcgn_cpu_send_0.cpp                     \
  tests/dcgn_cpu_recv_0.cpp                     \
  tests/dcgn_cpu_send_1.cpp                     \
  tests/dcgn_cpu_recv_1.cpp                     \
  tests/mpi_barrier.cpp                         \
  tests/mpi_cpu_send_0.cpp                      \
  tests/mpi_cpu_recv_0.cpp                      \
	tests/mpi_cpu_bcast_0.cpp											\
  tests/mpi_ping_cc.cpp                         \

TEST_CU_SRC=                                    \
  tests/dcgn_cg_barrier_0.cu                    \
  tests/dcgn_cg_broadcast_0.cu                  \
  tests/dcgn_cg_broadcast_1.cu                  \
  tests/dcgn_cg_ping_0.cu                       \
  tests/dcgn_cg_ping_1.cu                       \
  tests/dcgn_cg_recv_0.cu                       \
  tests/dcgn_cg_recv_1.cu                       \
  tests/dcgn_cg_send_0.cu                       \
  tests/dcgn_cg_send_1.cu                       \
  tests/dcgn_gc_ping_0.cu                       \
  tests/dcgn_gc_ping_1.cu                       \
  tests/dcgn_gc_recv_0.cu                       \
  tests/dcgn_gc_recv_1.cu                       \
  tests/dcgn_gc_send_0.cu                       \
  tests/dcgn_gc_send_1.cu                       \
  tests/dcgn_gpu_barrier_0.cu                   \
  tests/dcgn_gpu_broadcast_0.cu                 \
  tests/dcgn_gpu_ping_0.cu                      \
  tests/dcgn_gpu_ping_1.cu                      \
  tests/dcgn_gpu_recv_0.cu                      \
  tests/dcgn_gpu_recv_1.cu                      \
  tests/dcgn_gpu_send_0.cu                      \
  tests/dcgn_gpu_send_1.cu                      \


CXX_OBJ=$(API_CXX_SRC:.cpp=.o)  $(INF_CXX_SRC:.cpp=.o)
CU_OBJ =$(API_CU_SRC:.cu=.cu_o) $(INF_CU_SRC:.cu=.cu_o)
SAMPLE_CXX_OBJ=$(SAMPLE_SRC:.cpp=.o)
SAMPLE_CU_OBJ=$(SAMPLE_CU_SRC:.cu=.cu_o)
TEST_CPP_SRC=$(TEST_CPP_SRC:.cpp=.o)

ALL=                            \
    $(LIBRARY)                  \
    cpu_send_recv               \
    cpu_async_send_recv         \
    cpu_send_async_recv         \
    cpu_async_send_async_recv   \
    cpu_gpu_barrier             \
    cpu_gpu_send_recv           \
    cpu_gpu_broadcast           \
    gpu_async_send_async_recv   \
    parallel_cpu_mandelbrot     \
    parallel_cpu_matmult        \
    parallel_cpu_nbody          \
    parallel_gpu_mandelbrot     \
    parallel_gpu_mandelbrot2    \
    parallel_gpu_matmult        \
    parallel_gpu_matmult2       \
    parallel_gpu_nbody          \
    parallel_gpu_nbody2         \
    parallel_gpu_nbody3         \
    serial_cpu_mandelbrot       \
    serial_cpu_matmult          \
    serial_cpu_nbody            \
    serial_gpu_mandelbrot       \
    serial_gpu_matmult          \
    serial_gpu_nbody            \
    speed_test_cpu              \
    speed_test_cpu2             \
    speed_test_gpu              \
                                \
    tests/dcgn_cpu_barrier_0    \
    tests/dcgn_cpu_barrier_1    \
    tests/dcgn_cpu_broadcast_0  \
    tests/dcgn_cpu_broadcast_1  \
    tests/dcgn_cpu_send_0       \
    tests/dcgn_cpu_recv_0       \
    tests/dcgn_cpu_send_1       \
    tests/dcgn_cpu_recv_1       \
    tests/dcgn_ping_cc_0        \
    tests/dcgn_ping_cc_1        \
    tests/mpi_barrier           \
    tests/mpi_cpu_send_0        \
    tests/mpi_cpu_recv_0        \
		tests/mpi_cpu_bcast_0				\
    tests/mpi_ping_cc           \
                                \
    tests/dcgn_cg_barrier_0     \
    tests/dcgn_cg_broadcast_0   \
    tests/dcgn_cg_broadcast_1   \
    tests/dcgn_cg_ping_0        \
    tests/dcgn_cg_ping_1        \
    tests/dcgn_cg_recv_0        \
    tests/dcgn_cg_recv_1        \
    tests/dcgn_cg_send_0        \
    tests/dcgn_cg_send_1        \
    tests/dcgn_gc_ping_0        \
    tests/dcgn_gc_ping_1        \
    tests/dcgn_gc_recv_0        \
    tests/dcgn_gc_recv_1        \
    tests/dcgn_gc_send_0        \
    tests/dcgn_gc_send_1        \
    tests/dcgn_gpu_barrier_0    \
    tests/dcgn_gpu_broadcast_0  \
    tests/dcgn_gpu_ping_0       \
    tests/dcgn_gpu_ping_1       \
    tests/dcgn_gpu_recv_0       \
    tests/dcgn_gpu_recv_1       \
    tests/dcgn_gpu_send_0       \
    tests/dcgn_gpu_send_1       \

all: $(ALL)

cpu_send_recv: samples/cpu_send_recv.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/cpu_send_recv.o -o cpu_send_recv $(DCGN_LIBS)

cpu_async_send_recv: samples/cpu_async_send_recv.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/cpu_async_send_recv.o -o cpu_async_send_recv $(DCGN_LIBS)

cpu_send_async_recv: samples/cpu_send_async_recv.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/cpu_send_async_recv.o -o cpu_send_async_recv $(DCGN_LIBS)

cpu_async_send_async_recv: samples/cpu_async_send_async_recv.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/cpu_async_send_async_recv.o -o cpu_async_send_async_recv $(DCGN_LIBS)

cpu_gpu_barrier: samples/cpu_gpu_barrier.cu_o samples/cpu_gpu_barrier.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/cpu_gpu_barrier.cu_o samples/cpu_gpu_barrier.o -o cpu_gpu_barrier $(DCGN_LIBS)

cpu_gpu_send_recv: samples/cpu_gpu_send_recv.cu_o samples/cpu_gpu_send_recv.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/cpu_gpu_send_recv.cu_o samples/cpu_gpu_send_recv.o -o cpu_gpu_send_recv $(DCGN_LIBS)

cpu_gpu_broadcast: samples/cpu_gpu_broadcast.cu_o samples/cpu_gpu_broadcast.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/cpu_gpu_broadcast.cu_o samples/cpu_gpu_broadcast.o -o cpu_gpu_broadcast $(DCGN_LIBS)

gpu_async_send_async_recv: samples/gpu_async_send_async_recv.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/gpu_async_send_async_recv.cu_o -o gpu_async_send_async_recv $(DCGN_LIBS)

serial_cpu_matmult: samples/serial_cpu_matmult.o
	$(CXX) -Wall $(OPTFLAGS) samples/serial_cpu_matmult.o -o serial_cpu_matmult

serial_gpu_matmult: samples/serial_gpu_matmult.o samples/gpu_matmult.cu_o
	$(CXX) -Wall $(OPTFLAGS) samples/serial_gpu_matmult.o samples/gpu_matmult.cu_o -o serial_gpu_matmult $(CUDA_LIBS)

parallel_cpu_matmult: samples/parallel_cpu_matmult.o
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_cpu_matmult.o -o parallel_cpu_matmult $(MPI_LIBS)

parallel_gpu_matmult: samples/parallel_gpu_matmult.o samples/gpu_matmult.cu_o
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_gpu_matmult.o samples/gpu_matmult.cu_o -o parallel_gpu_matmult $(CUDA_MPI_LIBS)

parallel_gpu_matmult2: samples/parallel_gpu_matmult.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_gpu_matmult.cu_o -o parallel_gpu_matmult2 $(DCGN_LIBS)

serial_cpu_nbody: samples/serial_cpu_nbody.o
	$(CXX) -Wall $(OPTFLAGS) samples/serial_cpu_nbody.o -o serial_cpu_nbody

serial_gpu_nbody: samples/serial_gpu_nbody.cu_o
	$(CXX) -Wall $(OPTFLAGS) samples/serial_gpu_nbody.cu_o -o serial_gpu_nbody $(CUDA_LIBS)

parallel_cpu_nbody: samples/parallel_cpu_nbody.o
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_cpu_nbody.o -o parallel_cpu_nbody $(MPI_LIBS)

parallel_gpu_nbody: samples/parallel_gpu_nbody.cu_o
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_gpu_nbody.cu_o -o parallel_gpu_nbody  $(CUDA_MPI_LIBS)

parallel_gpu_nbody2: samples/parallel_gpu_nbody2.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_gpu_nbody2.cu_o -o parallel_gpu_nbody2  $(DCGN_LIBS)

parallel_gpu_nbody3: samples/parallel_gpu_nbody3.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_gpu_nbody3.cu_o -o parallel_gpu_nbody3  $(DCGN_LIBS)

speed_test_cpu: samples/speed_test_cpu.o
	$(CXX) -Wall $(OPTFLAGS) samples/speed_test_cpu.o -o speed_test_cpu $(MPI_LIBS)

speed_test_cpu2: samples/speed_test_cpu2.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/speed_test_cpu2.o -o speed_test_cpu2 $(DCGN_LIBS)

speed_test_gpu: samples/speed_test_gpu.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/speed_test_gpu.cu_o -o speed_test_gpu $(DCGN_LIBS)

serial_cpu_mandelbrot: samples/serial_cpu_mandelbrot.o
	$(CXX) -Wall $(OPTFLAGS) samples/serial_cpu_mandelbrot.o -o serial_cpu_mandelbrot

parallel_cpu_mandelbrot: samples/parallel_cpu_mandelbrot.o
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_cpu_mandelbrot.o -o parallel_cpu_mandelbrot $(MPI_LIBS)

serial_gpu_mandelbrot: samples/serial_gpu_mandelbrot.cu_o
	$(CXX) -Wall $(OPTFLAGS) samples/serial_gpu_mandelbrot.cu_o -o serial_gpu_mandelbrot $(CUDA_MPI_LIBS)

parallel_gpu_mandelbrot: samples/parallel_gpu_mandelbrot.cu_o
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_gpu_mandelbrot.cu_o -o parallel_gpu_mandelbrot $(CUDA_MPI_LIBS)

parallel_gpu_mandelbrot2: samples/parallel_gpu_mandelbrot2.cu_o  $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) samples/parallel_gpu_mandelbrot2.cu_o -o parallel_gpu_mandelbrot2 $(DCGN_LIBS)

tests/dcgn_cpu_barrier_0: tests/dcgn_cpu_barrier_0.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cpu_barrier_0.o -o tests/dcgn_cpu_barrier_0 $(DCGN_LIBS)

tests/dcgn_cpu_barrier_1: tests/dcgn_cpu_barrier_1.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cpu_barrier_1.o -o tests/dcgn_cpu_barrier_1 $(DCGN_LIBS)

tests/dcgn_cpu_broadcast_0: tests/dcgn_cpu_broadcast_0.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cpu_broadcast_0.o -o tests/dcgn_cpu_broadcast_0 $(DCGN_LIBS)

tests/dcgn_cpu_broadcast_1: tests/dcgn_cpu_broadcast_1.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cpu_broadcast_1.o -o tests/dcgn_cpu_broadcast_1 $(DCGN_LIBS)

tests/dcgn_cpu_send_0: tests/dcgn_cpu_send_0.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cpu_send_0.o -o tests/dcgn_cpu_send_0 $(DCGN_LIBS)

tests/dcgn_cpu_recv_0: tests/dcgn_cpu_recv_0.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cpu_recv_0.o -o tests/dcgn_cpu_recv_0 $(DCGN_LIBS)

tests/dcgn_cpu_send_1: tests/dcgn_cpu_send_1.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cpu_send_1.o -o tests/dcgn_cpu_send_1 $(DCGN_LIBS)

tests/dcgn_cpu_recv_1: tests/dcgn_cpu_recv_1.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cpu_recv_1.o -o tests/dcgn_cpu_recv_1 $(DCGN_LIBS)

tests/mpi_barrier: tests/mpi_barrier.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/mpi_barrier.o -o tests/mpi_barrier $(DCGN_LIBS)

tests/mpi_cpu_send_0: tests/mpi_cpu_send_0.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/mpi_cpu_send_0.o -o tests/mpi_cpu_send_0 $(DCGN_LIBS)

tests/mpi_cpu_recv_0: tests/mpi_cpu_recv_0.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/mpi_cpu_recv_0.o -o tests/mpi_cpu_recv_0 $(DCGN_LIBS)

tests/dcgn_cg_barrier_0: tests/dcgn_cg_barrier_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_barrier_0.cu_o -o tests/dcgn_cg_barrier_0 $(DCGN_LIBS)

tests/dcgn_cg_broadcast_0: tests/dcgn_cg_broadcast_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_broadcast_0.cu_o -o tests/dcgn_cg_broadcast_0 $(DCGN_LIBS)

tests/dcgn_cg_broadcast_1: tests/dcgn_cg_broadcast_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_broadcast_1.cu_o -o tests/dcgn_cg_broadcast_1 $(DCGN_LIBS)

tests/dcgn_cg_ping_0: tests/dcgn_cg_ping_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_ping_0.cu_o -o tests/dcgn_cg_ping_0 $(DCGN_LIBS)

tests/dcgn_cg_ping_1: tests/dcgn_cg_ping_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_ping_1.cu_o -o tests/dcgn_cg_ping_1 $(DCGN_LIBS)

tests/dcgn_cg_send_0: tests/dcgn_cg_send_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_send_0.cu_o -o tests/dcgn_cg_send_0 $(DCGN_LIBS)

tests/dcgn_cg_send_1: tests/dcgn_cg_send_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_send_1.cu_o -o tests/dcgn_cg_send_1 $(DCGN_LIBS)

tests/dcgn_cg_recv_0: tests/dcgn_cg_recv_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_recv_0.cu_o -o tests/dcgn_cg_recv_0 $(DCGN_LIBS)

tests/dcgn_cg_recv_1: tests/dcgn_cg_recv_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_cg_recv_1.cu_o -o tests/dcgn_cg_recv_1 $(DCGN_LIBS)

tests/dcgn_gc_ping_0: tests/dcgn_gc_ping_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gc_ping_0.cu_o -o tests/dcgn_gc_ping_0 $(DCGN_LIBS)

tests/dcgn_gc_ping_1: tests/dcgn_gc_ping_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gc_ping_1.cu_o -o tests/dcgn_gc_ping_1 $(DCGN_LIBS)

tests/dcgn_gc_recv_0: tests/dcgn_gc_recv_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gc_recv_0.cu_o -o tests/dcgn_gc_recv_0 $(DCGN_LIBS)

tests/dcgn_gc_recv_1: tests/dcgn_gc_recv_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gc_recv_1.cu_o -o tests/dcgn_gc_recv_1 $(DCGN_LIBS)

tests/dcgn_gc_send_0: tests/dcgn_gc_send_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gc_send_0.cu_o -o tests/dcgn_gc_send_0 $(DCGN_LIBS)

tests/dcgn_gc_send_1: tests/dcgn_gc_send_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gc_send_1.cu_o -o tests/dcgn_gc_send_1 $(DCGN_LIBS)

tests/dcgn_gpu_barrier_0: tests/dcgn_gpu_barrier_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gpu_barrier_0.cu_o -o tests/dcgn_gpu_barrier_0 $(DCGN_LIBS)

tests/dcgn_gpu_broadcast_0: tests/dcgn_gpu_broadcast_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gpu_broadcast_0.cu_o -o tests/dcgn_gpu_broadcast_0 $(DCGN_LIBS)

tests/dcgn_gpu_ping_0: tests/dcgn_gpu_ping_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gpu_ping_0.cu_o -o tests/dcgn_gpu_ping_0 $(DCGN_LIBS)

tests/dcgn_gpu_ping_1: tests/dcgn_gpu_ping_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gpu_ping_1.cu_o -o tests/dcgn_gpu_ping_1 $(DCGN_LIBS)

tests/dcgn_gpu_recv_0: tests/dcgn_gpu_recv_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gpu_recv_0.cu_o -o tests/dcgn_gpu_recv_0 $(DCGN_LIBS)

tests/dcgn_gpu_recv_1: tests/dcgn_gpu_recv_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gpu_recv_1.cu_o -o tests/dcgn_gpu_recv_1 $(DCGN_LIBS)

tests/dcgn_gpu_send_0: tests/dcgn_gpu_send_0.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gpu_send_0.cu_o -o tests/dcgn_gpu_send_0 $(DCGN_LIBS)

tests/dcgn_gpu_send_1: tests/dcgn_gpu_send_1.cu_o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_gpu_send_1.cu_o -o tests/dcgn_gpu_send_1 $(DCGN_LIBS)

tests/mpi_cpu_bcast_0: tests/mpi_cpu_bcast_0.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/mpi_cpu_bcast_0.o -o tests/mpi_cpu_bcast_0 $(DCGN_LIBS)

tests/mpi_ping_cc: tests/mpi_ping_cc.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/mpi_ping_cc.o -o tests/mpi_ping_cc $(DCGN_LIBS)

tests/dcgn_ping_cc_0: tests/dcgn_ping_cc_0.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_ping_cc_0.o -o tests/dcgn_ping_cc_0 $(DCGN_LIBS)

tests/dcgn_ping_cc_1: tests/dcgn_ping_cc_1.o $(LIBRARY)
	$(CXX) -Wall $(OPTFLAGS) tests/dcgn_ping_cc_1.o -o tests/dcgn_ping_cc_1 $(DCGN_LIBS)

$(LIBRARY): $(CXX_OBJ) $(CU_OBJ)
	$(AR) $(ARFLAGS) $(LIBRARY) $(CXX_OBJ) $(CU_OBJ)

%.cu_o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $*.cu_o  $(NVCC_CXXFLAGS)

output: ./serial_cpu_matmult ./serial_cpu_nbody
	./serial_cpu_matmult 8,8,8,samples/matrix00008.in0
	./serial_cpu_matmult 8,8,9,samples/matrix00008.in1
	./serial_cpu_matmult 16,16,16,samples/matrix00016.in0
	./serial_cpu_matmult 16,16,17,samples/matrix00016.in1
	./serial_cpu_matmult 32,32,32,samples/matrix00032.in0
	./serial_cpu_matmult 32,32,33,samples/matrix00032.in1
	./serial_cpu_matmult 256,256,256,samples/matrix00256.in0
	./serial_cpu_matmult 256,256,257,samples/matrix00256.in1
	./serial_cpu_matmult 1024,1024,1024,samples/matrix01024.in0
	./serial_cpu_matmult 1024,1024,1025,samples/matrix01024.in1
	./serial_cpu_matmult 2048,2048,2048,samples/matrix02048.in0
	./serial_cpu_matmult 2048,2048,2049,samples/matrix02048.in1
	./serial_cpu_matmult 4096,4096,4096,samples/matrix04096.in0
	./serial_cpu_matmult 4096,4096,4097,samples/matrix04096.in1
	./serial_cpu_nbody  1024 samples/nbody01024.in
	./serial_cpu_nbody  2048 samples/nbody02048.in
	./serial_cpu_nbody  4096 samples/nbody04096.in
	./serial_cpu_nbody  8192 samples/nbody08192.in
	./serial_cpu_nbody 16384 samples/nbody16384.in
	./serial_cpu_nbody 32768 samples/nbody32768.in
	./serial_cpu_nbody 65536 samples/nbody65536.in

clean:
	@rm -fv $(LIBRARY) $(CXX_OBJ) $(CU_OBJ) $(SAMPLE_CXX_OBJ) $(SAMPLE_CU_OBJ) $(ALL) samples/mandelbrot.pgm samples/mat*.in* samples/mat*.out samples/nbody*.in samples/nbody*.out tests/*.cu_o tests/*.o
