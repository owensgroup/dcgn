#include <mpi.h>
#include <algorithm>
#include <cstdio>

typedef double (* SpeedTest)(const int id, const int rank, const int param, char * const units);

double localSendTest(const int id, const int rank, const int param, char * const units);
double localSendRecvTest(const int id, const int rank, const int param, char * const units);

double globalSendTest(const int id, const int rank, const int param, char * const units);
double globalRecvTest(const int id, const int rank, const int param, char * const units);
double globalSendRecvTest(const int id, const int rank, const int param, char * const units);
double globalBarrierTest(const int id, const int rank, const int param, char * const units);
double globalBroadcastSendTest(const int id, const int rank, const int param, char * const units);
double globalBroadcastRecvTest(const int id, const int rank, const int param, char * const units);
double globalBroadcastRecv2Test(const int id, const int rank, const int param, char * const units);
double globalBroadcastRecv3Test(const int id, const int rank, const int param, char * const units);
double globalBroadcastRecv4Test(const int id, const int rank, const int param, char * const units);

int main(int argc, char ** argv)
{
  SpeedTest tests[] =
  {
    localSendTest,
    localSendRecvTest,

    globalSendTest,
    globalRecvTest,
    globalSendRecvTest,
    globalBarrierTest,
    globalBroadcastSendTest,
    globalBroadcastRecvTest,
    globalBroadcastRecv2Test,
    globalBroadcastRecv3Test,
    globalBroadcastRecv4Test,
  };
  const char * const testNames[] =
  {
    "Local Send",
    "Local SendRecvReplace",

    "Global Send",
    "Global Recv",
    "Global SendRecvReplace",
    "Global Barrier",
    "Global Broadcast Send",
    "Global Broadcast Recv",
    "Global Broadcast Recv 2",
    "Global Broadcast Recv 3",
    "Global Broadcast Recv 4",
  };
  const int NUM_TESTS = sizeof(tests) / sizeof(tests[0]);
  const int NUM_UNITS = 17;
  const int NUM_ITERS = 30;
  double runTimes[NUM_ITERS];

  int id, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < NUM_TESTS; ++i)
  {
    if (id == 0)
    {
      printf("%-25s - %3d CPUs.\n", testNames[i], size);
      printf("           | Min. Time   | Max. Time   | Mean Time   | Median Time |\n");
      printf("+----------+-------------+-------------+-------------+-------------+\n");
    }
    char units[40];
    for (int j = 0; j < NUM_UNITS; ++j)
    {
      for (int k = 0; k < NUM_ITERS; ++k)
      {
        runTimes[k] = tests[i](id, size, j, units);
      }
      std::sort(runTimes, runTimes + NUM_ITERS);
      double avg = 0.0;
      for (int k = 0; k < NUM_ITERS; ++k) avg += runTimes[k];
      avg /= NUM_ITERS;
      if (id == 0)
      {
        printf("| %-8s | %11f | %11f | %11f | %11f |\n", units, runTimes[0], runTimes[NUM_ITERS - 1], avg, runTimes[NUM_ITERS / 2]);
      }
    }
    if (id == 0)
    {
      printf("+----------+-------------+-------------+-------------+-------------+\n");
    }
  }
  MPI_Finalize();
  return 0;
}

double localSendTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 0)
  {
    MPI_Status stat;
    MPI_Request req1, req2;
    char * buf = new char[amt];
    double t = MPI_Wtime();
    MPI_Irecv(buf, amt, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req1);
    MPI_Isend(buf, amt, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req2);
    MPI_Wait(&req1, &stat);
    MPI_Wait(&req2, &stat);
    t = MPI_Wtime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double localSendRecvTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 0)
  {
    MPI_Status stat;
    char * buf = new char[amt];
    double t = MPI_Wtime();
    MPI_Sendrecv_replace(buf, amt, MPI_BYTE, 0, 0, 0, 0, MPI_COMM_WORLD, &stat);
    t = MPI_Wtime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double localBarrierTest(const int id, const int rank, const int param, char * const units)
{
  const int TABLE[] =
  {
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
  };
  sprintf(units, "%d", TABLE[param]);
  if (id == 0)
  {
    double t = MPI_Wtime();
    for (int i = TABLE[param]; i != 0; --i)
    {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    return MPI_Wtime() - t;
  }
  return -1.0;
}

double localBroadcastTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 0)
  {
    char * buf = new char[amt];
    double t = MPI_Wtime();
    MPI_Bcast(buf, amt, MPI_BYTE, 0, MPI_COMM_WORLD);
    return MPI_Wtime() - t;
  }
  return -1.0;
}


double globalSendTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 0)
  {
    char * buf = new char[amt];
    double t = MPI_Wtime();
    MPI_Send(buf, amt, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    t = MPI_Wtime() - t;
    delete [] buf;
    return t;
  }
  else if (id == 1)
  {
    MPI_Status stat;
    char * buf = new char[amt];
    double t = MPI_Wtime();
    MPI_Recv(buf, amt, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &stat);
    t = MPI_Wtime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double globalRecvTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 1)
  {
    char * buf = new char[amt];
    double t = MPI_Wtime();
    MPI_Send(buf, amt, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    t = MPI_Wtime() - t;
    delete [] buf;
    return t;
  }
  else if (id == 0)
  {
    MPI_Status stat;
    char * buf = new char[amt];
    double t = MPI_Wtime();
    MPI_Recv(buf, amt, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &stat);
    t = MPI_Wtime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double globalSendRecvTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);
  if (id == 0 || id == 1)
  {
    MPI_Status stat;
    char * buf = new char[amt];
    double t = MPI_Wtime();
    MPI_Sendrecv_replace(buf, amt, MPI_BYTE, 1 - id, 0, 1 - id, 0, MPI_COMM_WORLD, &stat);
    t = MPI_Wtime() - t;
    delete [] buf;
    return t;
  }
  return -1.0;
}

double globalBarrierTest(const int id, const int rank, const int param, char * const units)
{
  const int TABLE[] =
  {
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
  };
  sprintf(units, "%d", TABLE[param]);
  double t = MPI_Wtime();
  for (int i = TABLE[param]; i != 0; --i)
  {
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return MPI_Wtime() - t;
  return -1.0;
}

double globalBroadcastSendTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = MPI_Wtime();
  MPI_Bcast(buf, amt, MPI_BYTE, 0, MPI_COMM_WORLD);
  return MPI_Wtime() - t;

  return -1.0;
}

double globalBroadcastRecvTest(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = MPI_Wtime();
  MPI_Bcast(buf, amt, MPI_BYTE, 1, MPI_COMM_WORLD);
  return MPI_Wtime() - t;

  return -1.0;
}

double globalBroadcastRecv2Test(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = MPI_Wtime();
  MPI_Bcast(buf, amt, MPI_BYTE, 2, MPI_COMM_WORLD);
  return MPI_Wtime() - t;

  return -1.0;
}

double globalBroadcastRecv3Test(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = MPI_Wtime();
  MPI_Bcast(buf, amt, MPI_BYTE, 3, MPI_COMM_WORLD);
  return MPI_Wtime() - t;

  return -1.0;
}
double globalBroadcastRecv4Test(const int id, const int rank, const int param, char * const units)
{
  const int amt = 1024 << param;
  if      (amt < 1048576)         sprintf(units, "%d kB", amt / 1024);
  else if (amt < 1048576 * 1024)  sprintf(units, "%d MB", amt / 1024 / 1024);

  char * buf = new char[amt];
  double t = MPI_Wtime();
  MPI_Bcast(buf, amt, MPI_BYTE, 6, MPI_COMM_WORLD);
  return MPI_Wtime() - t;

  return -1.0;
}

