#include <iostream>
#include <vector>

#define CHECK_ERROR(call)                                                \
  do {                                                                   \
    int _error = (call);                                                 \
    if (_error) {                                                        \
      printf("*** Error *** at [%s:%d] error=%d \n", __FILE__, __LINE__, \
             _error);                                                    \
    }                                                                    \
  } while (0)

#define CUDA_CHECK_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t _error = (cudaError_t)(call);                                  \
    if (_error != cudaSuccess) {                                               \
      printf("*** CUDA Error *** at [%s:%d] error=%d, reason:%s \n", __FILE__, \
             __LINE__, _error, cudaGetErrorString(_error));                    \
    }                                                                          \
  } while (0)

void setValue(std::vector<float> &array, int size) {
  for (int i = 0; i < size; i++) {
    array[i] = 1;
  }
}

void cpuNaive(float *A, float *B, float *C, int M, int N, int K) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      for (int i = 0; i < N; i++) {
        C[r * i + c] = A[r * N + i] * B[i * K + c];
      }
    }
  }
}

// DRAM Frequency         cycle/nsecond             6.80
// SM Frequency           cycle/nsecond             1.37
// Elapsed Cycles         cycle                     164,876
// Memory [%]             %                         0.42
// SOL DRAM               %                         0.42
// Duration               usecond                   120.70
// SOL L1/TEX Cache       %                         10.05
// SOL L2 Cache           %                         0.13
// SM Active Cycles       cycle                     2,398.35
// SM [%]                  %                        0.15
__global__ void sgemmNaive(float *A, float *B, float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float c = 0;
  if ((row < M) && (col < K)) {
    for (int i = 0; i < N; i++) {
      c += A[row * N + i] * B[i * K + col];
    }
    C[row * K + col] = c;
  }
}

// DRAM Frequency         cycle/nsecond             6.75
// SM Frequency           cycle/nsecond             1.36
// Elapsed Cycles         cycle                     871,222,338
// Memory [%]             %                         31.23
// SOL DRAM               %                         31.23
// Duration               msecond                   642.37
// SOL L1/TEX Cache       %                         29.40
// SOL L2 Cache           %                         12.67
// SM Active Cycles       cycle                     879,553,492.93
// SM [%]                  %                        3.07
// till array into several parts, each thread block computes one part
template <int row, int column>
__global__ void sgemmTilling(float *A, float *B, float *C, int M, int N,
                             int K) {
  __shared__ float aPart[row][column];
  __shared__ float bPart[column][row];

  int iter = N / blockDim.y;
  float tempResults[row][column] = {0};

  for (int i = 0; i < iter; i++) {
    // read part of array A and B into shared memory
    int idx = threadIdx.x * N + threadIdx.y + i * blockDim.y;
    aPart[threadIdx.x][threadIdx.y] = A[idx];
    __syncthreads();

    idx = (threadIdx.x + i * blockDim.x) * K + threadIdx.y;
    bPart[threadIdx.x][threadIdx.y] = B[idx];
    __syncthreads();

    // compute part sum
    tempResults[threadIdx.x][threadIdx.y] +=
        aPart[threadIdx.x][threadIdx.y] * bPart[threadIdx.y][threadIdx.x];
    __syncthreads();
  }

  // write results to global memory
  int globalIdx = (blockIdx.x * blockDim.x + threadIdx.x) * K + threadIdx.y;
  C[globalIdx] = tempResults[threadIdx.x][threadIdx.y];
}

// DRAM Frequency         cycle/nsecond             6.79
// SM Frequency           cycle/nsecond             1.36
// Elapsed Cycles         cycle                     81,627,029
// Memory [%]             %                         48.47
// SOL DRAM               %                         2.84
// Duration               msecond                   59.80
// SOL L1/TEX Cache       %                         96.93
// SOL L2 Cache           %                         3.97
// SM Active Cycles       cycle                     79,055,325.84
// SM [%]                  %                        17.15
__global__ void sgemmTillingV2(float *A, float *B, float *C, int M, int N,
                               int K) {
  __shared__ float tempA[128][8];
  __shared__ float tempB[8][128];

  int rowA = blockIdx.x;
  int colB = blockIdx.y;
  int threadId = threadIdx.x * blockDim.y + threadIdx.y;

  int wrapId = threadId >> 5;
  int laneId = threadId & 31;
  // map wrapId to wrapResults row and column
  int wrapRow = wrapId / 2;
  int wrapCol = wrapId % 2;
  // map laneId to thread level resuls row and column
  int threadRow = laneId / 8;
  int threadCol = laneId % 8;

  // registers to save temp results
  float threadResults[8][8] = {0};
  float4 A1, A2, B1, B2;

  int iter = N / 8;
  for (int i = 0; i < iter; i++) {
    // read 128x8 data into tempA
    int row, col;
    if (threadId < 128) {
      row = (rowA * 128 + threadId);
      col = i * 8;
      float *tempBasePtr = &(tempA[threadId][0]);
      *(reinterpret_cast<float4 *>(tempBasePtr)) =
          *(reinterpret_cast<float4 *>(A + row * N + col));

      *(reinterpret_cast<float4 *>(tempBasePtr + 4)) =
          *(reinterpret_cast<float4 *>(A + row * N + col + 4));
    } else  // read 128x8 data into tempB
    {
      row = i * 8;
      col = colB * 128 + threadId - 128;
      tempB[0][threadId - 128] = *(B + row * K + col);
      tempB[1][threadId - 128] = *(B + (row + 1) * K + col);
      tempB[2][threadId - 128] = *(B + (row + 2) * K + col);
      tempB[3][threadId - 128] = *(B + (row + 3) * K + col);
      tempB[4][threadId - 128] = *(B + (row + 4) * K + col);
      tempB[5][threadId - 128] = *(B + (row + 5) * K + col);
      tempB[6][threadId - 128] = *(B + (row + 6) * K + col);
      tempB[7][threadId - 128] = *(B + (row + 7) * K + col);
    }
    __syncthreads();

    // perform wrap level computing

    for (int i = 0; i < 8; i++) {
      A1.x = tempA[wrapRow * 32 + threadRow * 4][i];
      A1.y = tempA[wrapRow * 32 + threadRow * 4 + 1][i];
      A1.z = tempA[wrapRow * 32 + threadRow * 4 + 2][i];
      A1.w = tempA[wrapRow * 32 + threadRow * 4 + 3][i];

      A1.x = tempA[wrapRow * 32 + threadRow * 4 + 4][i];
      A2.y = tempA[wrapRow * 32 + threadRow * 4 + 5][i];
      A2.z = tempA[wrapRow * 32 + threadRow * 4 + 6][i];
      A2.w = tempA[wrapRow * 32 + threadRow * 4 + 7][i];

      B1.x = tempB[i][wrapCol * 64 + threadCol * 4];
      B1.y = tempB[i][wrapCol * 64 + threadCol * 4 + 1];
      B1.z = tempB[i][wrapCol * 64 + threadCol * 4 + 2];
      B1.w = tempB[i][wrapCol * 64 + threadCol * 4 + 3];

      B2.x = tempB[i][wrapCol * 64 + threadCol * 4 + 8];
      B2.y = tempB[i][wrapCol * 64 + threadCol * 4 + 9];
      B2.z = tempB[i][wrapCol * 64 + threadCol * 4 + 10];
      B2.w = tempB[i][wrapCol * 64 + threadCol * 4 + 11];

      // fisrt 4x4
      threadResults[0][0] += A1.x * B1.x;
      threadResults[0][1] += A1.x * B1.y;
      threadResults[0][2] += A1.x * B1.z;
      threadResults[0][3] += A1.x * B1.w;
      threadResults[1][0] += A1.y * B1.x;
      threadResults[1][1] += A1.y * B1.y;
      threadResults[1][2] += A1.y * B1.z;
      threadResults[1][3] += A1.y * B1.w;
      threadResults[2][0] += A1.z * B1.x;
      threadResults[2][1] += A1.z * B1.y;
      threadResults[2][2] += A1.z * B1.z;
      threadResults[2][3] += A1.z * B1.w;
      threadResults[3][0] += A1.w * B1.x;
      threadResults[3][1] += A1.w * B1.y;
      threadResults[3][2] += A1.w * B1.z;
      threadResults[3][3] += A1.w * B1.w;

      // second 4x4
      threadResults[0][4] += A1.x * B2.x;
      threadResults[0][5] += A1.x * B2.y;
      threadResults[0][6] += A1.x * B2.z;
      threadResults[0][7] += A1.x * B2.w;
      threadResults[1][4] += A1.y * B2.x;
      threadResults[1][5] += A1.y * B2.y;
      threadResults[1][6] += A1.y * B2.z;
      threadResults[1][7] += A1.y * B2.w;
      threadResults[2][4] += A1.z * B2.x;
      threadResults[2][5] += A1.z * B2.y;
      threadResults[2][6] += A1.z * B2.z;
      threadResults[2][7] += A1.z * B2.w;
      threadResults[3][4] += A1.w * B2.x;
      threadResults[3][5] += A1.w * B2.y;
      threadResults[3][6] += A1.w * B2.z;
      threadResults[3][7] += A1.w * B2.w;

      // third 4x4
      threadResults[4][0] += A2.x * B1.x;
      threadResults[4][1] += A2.x * B1.y;
      threadResults[4][2] += A2.x * B1.z;
      threadResults[4][3] += A2.x * B1.w;
      threadResults[5][0] += A2.y * B1.x;
      threadResults[5][1] += A2.y * B1.y;
      threadResults[5][2] += A2.y * B1.z;
      threadResults[5][3] += A2.y * B1.w;
      threadResults[6][0] += A2.z * B1.x;
      threadResults[6][1] += A2.z * B1.y;
      threadResults[6][2] += A2.z * B1.z;
      threadResults[6][3] += A2.z * B1.w;
      threadResults[7][0] += A2.w * B1.x;
      threadResults[7][1] += A2.w * B1.y;
      threadResults[7][2] += A2.w * B1.z;
      threadResults[7][3] += A2.w * B1.w;

      // last 4x4
      threadResults[4][4] += A2.x * B2.x;
      threadResults[4][5] += A2.x * B2.y;
      threadResults[4][6] += A2.x * B2.z;
      threadResults[4][7] += A2.x * B2.w;
      threadResults[5][4] += A2.y * B2.x;
      threadResults[5][5] += A2.y * B2.y;
      threadResults[5][6] += A2.y * B2.z;
      threadResults[5][7] += A2.y * B2.w;
      threadResults[6][4] += A2.z * B2.x;
      threadResults[6][5] += A2.z * B2.y;
      threadResults[6][6] += A2.z * B2.z;
      threadResults[6][7] += A2.z * B2.w;
      threadResults[7][4] += A2.w * B2.x;
      threadResults[7][5] += A2.w * B2.y;
      threadResults[7][6] += A2.w * B2.z;
      threadResults[7][7] += A2.w * B2.w;
    }
  }

  // write temp thread level results back to global memory, each thread wirtes
  // 8x8 elements
  int globalRow = blockIdx.x * 128 + wrapRow * 32 + threadRow * 4;
  int globalCol = blockIdx.y * 128 + wrapCol * 64 + threadCol * 4;

  // first row
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])));
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])) + 1);
  // second row
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])) + 1);
  // third row
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])) + 1);
  // fourth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])) + 1);
  // fiveth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])) + 1);
  // sixth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])) + 1);
  // seventh row
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])) + 1);
  // last row
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])) + 1);
}

// DRAM Frequency         cycle/nsecond             6.79
// SM Frequency           cycle/nsecond             1.36
// Elapsed Cycles         cycle                     53,471.281
// Memory [%]             %                         48.15
// SOL DRAM               %                         4.90
// Duration               msecond                   39.17
// SOL L1/TEX Cache       %                         96.30
// SOL L2 Cache           %                         6.06
// SM Active Cycles       cycle                     41,433,151.38
// SM [%]                  %                        26.07
// add pragma unroll direvtive
__global__ void sgemmTillingV2_5(float *A, float *B, float *C, int M, int N,
                                 int K) {
  __shared__ float tempA[128][8];
  __shared__ float tempB[8][128];

  int rowA = blockIdx.x;
  int colB = blockIdx.y;
  int threadId = threadIdx.x * blockDim.y + threadIdx.y;

  int wrapId = threadId >> 5;
  int laneId = threadId & 31;
  // map wrapId to wrapResults row and column
  int wrapRow = wrapId / 2;
  int wrapCol = wrapId % 2;
  // map laneId to thread level resuls row and column
  int threadRow = laneId / 8;
  int threadCol = laneId % 8;

  // registers to save temp results
  float threadResults[8][8] = {0};
  float4 A1, A2, B1, B2;

  int iter = N / 8;
  for (int i = 0; i < iter; i++) {
    // read 128x8 data into tempA
    int row, col;
    if (threadId < 128) {
      row = (rowA * 128 + threadId);
      col = i * 8;
      float *tempBasePtr = &(tempA[threadId][0]);
      *(reinterpret_cast<float4 *>(tempBasePtr)) =
          *(reinterpret_cast<float4 *>(A + row * N + col));

      *(reinterpret_cast<float4 *>(tempBasePtr + 4)) =
          *(reinterpret_cast<float4 *>(A + row * N + col + 4));
    } else  // read 128x8 data into tempB
    {
      row = i * 8;
      col = colB * 128 + threadId - 128;
      tempB[0][threadId - 128] = *(B + row * K + col);
      tempB[1][threadId - 128] = *(B + (row + 1) * K + col);
      tempB[2][threadId - 128] = *(B + (row + 2) * K + col);
      tempB[3][threadId - 128] = *(B + (row + 3) * K + col);
      tempB[4][threadId - 128] = *(B + (row + 4) * K + col);
      tempB[5][threadId - 128] = *(B + (row + 5) * K + col);
      tempB[6][threadId - 128] = *(B + (row + 6) * K + col);
      tempB[7][threadId - 128] = *(B + (row + 7) * K + col);
    }
    __syncthreads();

// perform wrap level computing
#pragma unroll
    for (int i = 0; i < 8; i++) {
      A1.x = tempA[wrapRow * 32 + threadRow * 4][i];
      A1.y = tempA[wrapRow * 32 + threadRow * 4 + 1][i];
      A1.z = tempA[wrapRow * 32 + threadRow * 4 + 2][i];
      A1.w = tempA[wrapRow * 32 + threadRow * 4 + 3][i];

      A1.x = tempA[wrapRow * 32 + threadRow * 4 + 4][i];
      A2.y = tempA[wrapRow * 32 + threadRow * 4 + 5][i];
      A2.z = tempA[wrapRow * 32 + threadRow * 4 + 6][i];
      A2.w = tempA[wrapRow * 32 + threadRow * 4 + 7][i];

      B1.x = tempB[i][wrapCol * 64 + threadCol * 4];
      B1.y = tempB[i][wrapCol * 64 + threadCol * 4 + 1];
      B1.z = tempB[i][wrapCol * 64 + threadCol * 4 + 2];
      B1.w = tempB[i][wrapCol * 64 + threadCol * 4 + 3];

      B2.x = tempB[i][wrapCol * 64 + threadCol * 4 + 8];
      B2.y = tempB[i][wrapCol * 64 + threadCol * 4 + 9];
      B2.z = tempB[i][wrapCol * 64 + threadCol * 4 + 10];
      B2.w = tempB[i][wrapCol * 64 + threadCol * 4 + 11];

      // fisrt 4x4
      threadResults[0][0] += A1.x * B1.x;
      threadResults[0][1] += A1.x * B1.y;
      threadResults[0][2] += A1.x * B1.z;
      threadResults[0][3] += A1.x * B1.w;
      threadResults[1][0] += A1.y * B1.x;
      threadResults[1][1] += A1.y * B1.y;
      threadResults[1][2] += A1.y * B1.z;
      threadResults[1][3] += A1.y * B1.w;
      threadResults[2][0] += A1.z * B1.x;
      threadResults[2][1] += A1.z * B1.y;
      threadResults[2][2] += A1.z * B1.z;
      threadResults[2][3] += A1.z * B1.w;
      threadResults[3][0] += A1.w * B1.x;
      threadResults[3][1] += A1.w * B1.y;
      threadResults[3][2] += A1.w * B1.z;
      threadResults[3][3] += A1.w * B1.w;

      // second 4x4
      threadResults[0][4] += A1.x * B2.x;
      threadResults[0][5] += A1.x * B2.y;
      threadResults[0][6] += A1.x * B2.z;
      threadResults[0][7] += A1.x * B2.w;
      threadResults[1][4] += A1.y * B2.x;
      threadResults[1][5] += A1.y * B2.y;
      threadResults[1][6] += A1.y * B2.z;
      threadResults[1][7] += A1.y * B2.w;
      threadResults[2][4] += A1.z * B2.x;
      threadResults[2][5] += A1.z * B2.y;
      threadResults[2][6] += A1.z * B2.z;
      threadResults[2][7] += A1.z * B2.w;
      threadResults[3][4] += A1.w * B2.x;
      threadResults[3][5] += A1.w * B2.y;
      threadResults[3][6] += A1.w * B2.z;
      threadResults[3][7] += A1.w * B2.w;

      // third 4x4
      threadResults[4][0] += A2.x * B1.x;
      threadResults[4][1] += A2.x * B1.y;
      threadResults[4][2] += A2.x * B1.z;
      threadResults[4][3] += A2.x * B1.w;
      threadResults[5][0] += A2.y * B1.x;
      threadResults[5][1] += A2.y * B1.y;
      threadResults[5][2] += A2.y * B1.z;
      threadResults[5][3] += A2.y * B1.w;
      threadResults[6][0] += A2.z * B1.x;
      threadResults[6][1] += A2.z * B1.y;
      threadResults[6][2] += A2.z * B1.z;
      threadResults[6][3] += A2.z * B1.w;
      threadResults[7][0] += A2.w * B1.x;
      threadResults[7][1] += A2.w * B1.y;
      threadResults[7][2] += A2.w * B1.z;
      threadResults[7][3] += A2.w * B1.w;

      // last 4x4
      threadResults[4][4] += A2.x * B2.x;
      threadResults[4][5] += A2.x * B2.y;
      threadResults[4][6] += A2.x * B2.z;
      threadResults[4][7] += A2.x * B2.w;
      threadResults[5][4] += A2.y * B2.x;
      threadResults[5][5] += A2.y * B2.y;
      threadResults[5][6] += A2.y * B2.z;
      threadResults[5][7] += A2.y * B2.w;
      threadResults[6][4] += A2.z * B2.x;
      threadResults[6][5] += A2.z * B2.y;
      threadResults[6][6] += A2.z * B2.z;
      threadResults[6][7] += A2.z * B2.w;
      threadResults[7][4] += A2.w * B2.x;
      threadResults[7][5] += A2.w * B2.y;
      threadResults[7][6] += A2.w * B2.z;
      threadResults[7][7] += A2.w * B2.w;
    }
  }

  // write temp thread level results back to global memory, each thread wirtes
  // 8x8 elements
  int globalRow = blockIdx.x * 128 + wrapRow * 32 + threadRow * 4;
  int globalCol = blockIdx.y * 128 + wrapCol * 64 + threadCol * 4;

  // first row
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])));
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])) + 1);
  // second row
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])) + 1);
  // third row
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])) + 1);
  // fourth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])) + 1);
  // fiveth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])) + 1);
  // sixth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])) + 1);
  // seventh row
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])) + 1);
  // last row
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])) + 1);
}

// DRAM Frequency         cycle/nsecond             6.80
// SM Frequency           cycle/nsecond             1.37
// Elapsed Cycles         cycle                     22,482,068
// Memory [%]             %                         46.39
// SOL DRAM               %                         10.20
// Duration               msecond                   16.46
// SOL L1/TEX Cache       %                         92.78
// SOL L2 Cache           %                         14.41
// SM Active Cycles       cycle                     21,338,911,46
// SM [%]                  %                        61.76
// use one dimensional block to reduce calculation related to wrap
__global__ void sgemmTillingV3(float *A, float *B, float *C, int M, int N,
                               int K) {
  __shared__ float tempA[128][8];
  __shared__ float tempB[8][128];

  int rowA = blockIdx.x;
  int colB = blockIdx.y;
  int threadId = threadIdx.x;

  int wrapId = threadId >> 5;
  int laneId = threadId & 31;
  // map wrapId to wrapResults row and column
  int wrapRow = wrapId / 2;
  int wrapCol = wrapId % 2;
  // map laneId to thread level resuls row and column
  int threadRow = laneId / 8;
  int threadCol = laneId % 8;

  // registers to save temp results
  float threadResults[8][8] = {0};
  float4 A1, A2, B1, B2;

  int iter = N / 8;
  for (int i = 0; i < iter; i++) {
    // read 128x8 data into tempA
    int row, col;
    if (threadId < 128) {
      row = (rowA * 128 + threadId);
      col = i * 8;
      float *tempBasePtr = &(tempA[threadId][0]);
      *(reinterpret_cast<float4 *>(tempBasePtr)) =
          *(reinterpret_cast<float4 *>(A + row * N + col));

      *(reinterpret_cast<float4 *>(tempBasePtr + 4)) =
          *(reinterpret_cast<float4 *>(A + row * N + col + 4));
    } else  // read 128x8 data into tempB
    {
      row = i * 8;
      col = colB * 128 + threadId - 128;
      tempB[0][threadId - 128] = *(B + row * K + col);
      tempB[1][threadId - 128] = *(B + (row + 1) * K + col);
      tempB[2][threadId - 128] = *(B + (row + 2) * K + col);
      tempB[3][threadId - 128] = *(B + (row + 3) * K + col);
      tempB[4][threadId - 128] = *(B + (row + 4) * K + col);
      tempB[5][threadId - 128] = *(B + (row + 5) * K + col);
      tempB[6][threadId - 128] = *(B + (row + 6) * K + col);
      tempB[7][threadId - 128] = *(B + (row + 7) * K + col);
    }
    __syncthreads();

// perform wrap level computing
#pragma unroll
    for (int i = 0; i < 8; i++) {
      A1.x = tempA[wrapRow * 32 + threadRow * 4][i];
      A1.y = tempA[wrapRow * 32 + threadRow * 4 + 1][i];
      A1.z = tempA[wrapRow * 32 + threadRow * 4 + 2][i];
      A1.w = tempA[wrapRow * 32 + threadRow * 4 + 3][i];

      A1.x = tempA[wrapRow * 32 + threadRow * 4 + 4][i];
      A2.y = tempA[wrapRow * 32 + threadRow * 4 + 5][i];
      A2.z = tempA[wrapRow * 32 + threadRow * 4 + 6][i];
      A2.w = tempA[wrapRow * 32 + threadRow * 4 + 7][i];

      B1.x = tempB[i][wrapCol * 64 + threadCol * 4];
      B1.y = tempB[i][wrapCol * 64 + threadCol * 4 + 1];
      B1.z = tempB[i][wrapCol * 64 + threadCol * 4 + 2];
      B1.w = tempB[i][wrapCol * 64 + threadCol * 4 + 3];

      B2.x = tempB[i][wrapCol * 64 + threadCol * 4 + 8];
      B2.y = tempB[i][wrapCol * 64 + threadCol * 4 + 9];
      B2.z = tempB[i][wrapCol * 64 + threadCol * 4 + 10];
      B2.w = tempB[i][wrapCol * 64 + threadCol * 4 + 11];

      // fisrt 4x4
      threadResults[0][0] += A1.x * B1.x;
      threadResults[0][1] += A1.x * B1.y;
      threadResults[0][2] += A1.x * B1.z;
      threadResults[0][3] += A1.x * B1.w;
      threadResults[1][0] += A1.y * B1.x;
      threadResults[1][1] += A1.y * B1.y;
      threadResults[1][2] += A1.y * B1.z;
      threadResults[1][3] += A1.y * B1.w;
      threadResults[2][0] += A1.z * B1.x;
      threadResults[2][1] += A1.z * B1.y;
      threadResults[2][2] += A1.z * B1.z;
      threadResults[2][3] += A1.z * B1.w;
      threadResults[3][0] += A1.w * B1.x;
      threadResults[3][1] += A1.w * B1.y;
      threadResults[3][2] += A1.w * B1.z;
      threadResults[3][3] += A1.w * B1.w;

      // second 4x4
      threadResults[0][4] += A1.x * B2.x;
      threadResults[0][5] += A1.x * B2.y;
      threadResults[0][6] += A1.x * B2.z;
      threadResults[0][7] += A1.x * B2.w;
      threadResults[1][4] += A1.y * B2.x;
      threadResults[1][5] += A1.y * B2.y;
      threadResults[1][6] += A1.y * B2.z;
      threadResults[1][7] += A1.y * B2.w;
      threadResults[2][4] += A1.z * B2.x;
      threadResults[2][5] += A1.z * B2.y;
      threadResults[2][6] += A1.z * B2.z;
      threadResults[2][7] += A1.z * B2.w;
      threadResults[3][4] += A1.w * B2.x;
      threadResults[3][5] += A1.w * B2.y;
      threadResults[3][6] += A1.w * B2.z;
      threadResults[3][7] += A1.w * B2.w;

      // third 4x4
      threadResults[4][0] += A2.x * B1.x;
      threadResults[4][1] += A2.x * B1.y;
      threadResults[4][2] += A2.x * B1.z;
      threadResults[4][3] += A2.x * B1.w;
      threadResults[5][0] += A2.y * B1.x;
      threadResults[5][1] += A2.y * B1.y;
      threadResults[5][2] += A2.y * B1.z;
      threadResults[5][3] += A2.y * B1.w;
      threadResults[6][0] += A2.z * B1.x;
      threadResults[6][1] += A2.z * B1.y;
      threadResults[6][2] += A2.z * B1.z;
      threadResults[6][3] += A2.z * B1.w;
      threadResults[7][0] += A2.w * B1.x;
      threadResults[7][1] += A2.w * B1.y;
      threadResults[7][2] += A2.w * B1.z;
      threadResults[7][3] += A2.w * B1.w;

      // last 4x4
      threadResults[4][4] += A2.x * B2.x;
      threadResults[4][5] += A2.x * B2.y;
      threadResults[4][6] += A2.x * B2.z;
      threadResults[4][7] += A2.x * B2.w;
      threadResults[5][4] += A2.y * B2.x;
      threadResults[5][5] += A2.y * B2.y;
      threadResults[5][6] += A2.y * B2.z;
      threadResults[5][7] += A2.y * B2.w;
      threadResults[6][4] += A2.z * B2.x;
      threadResults[6][5] += A2.z * B2.y;
      threadResults[6][6] += A2.z * B2.z;
      threadResults[6][7] += A2.z * B2.w;
      threadResults[7][4] += A2.w * B2.x;
      threadResults[7][5] += A2.w * B2.y;
      threadResults[7][6] += A2.w * B2.z;
      threadResults[7][7] += A2.w * B2.w;
    }
  }

  // write temp thread level results back to global memory, each thread wirtes
  // 8x8 elements
  int globalRow = blockIdx.x * 128 + wrapRow * 32 + threadRow * 4;
  int globalCol = blockIdx.y * 128 + wrapCol * 64 + threadCol * 4;

  // first row
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])));
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])) + 1);
  // second row
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])) + 1);
  // third row
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])) + 1);
  // fourth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])) + 1);
  // fiveth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])) + 1);
  // sixth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])) + 1);
  // seventh row
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])) + 1);
  // last row
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])) + 1);
}

// replace mul operation by register access
__global__ void sgemmTillingV3_5(float *A, float *B, float *C, int M, int N,
                                 int K) {
  __shared__ float tempA[128][8];
  __shared__ float tempB[8][128];

  int rowA = blockIdx.x;
  int colB = blockIdx.y;
  int threadId = threadIdx.x;

  int wrapId = threadId >> 5;
  int laneId = threadId & 31;
  // map wrapId to wrapResults row and column
  int wrapRow32 = (wrapId / 2) << 5;
  int wrapCol64 = (wrapId % 2) << 6;
  // map laneId to thread level resuls row and column
  int threadRow4 = (laneId / 8) << 2;
  int threadCol4 = (laneId % 8) << 2;

  // registers to save temp results
  float threadResults[8][8] = {0};
  float4 A1, A2, B1, B2;

  int iter = N / 8;
  for (int i = 0; i < iter; i++) {
    // read 128x8 data into tempA
    int row, col;
    if (threadId < 128) {
      row = (rowA * 128 + threadId);
      col = i * 8;
      float *tempBasePtr = &(tempA[threadId][0]);
      *(reinterpret_cast<float4 *>(tempBasePtr)) =
          *(reinterpret_cast<float4 *>(A + row * N + col));

      *(reinterpret_cast<float4 *>(tempBasePtr + 4)) =
          *(reinterpret_cast<float4 *>(A + row * N + col + 4));
    } else  // read 128x8 data into tempB
    {
      row = i * 8;
      col = colB * 128 + threadId - 128;
      tempB[0][threadId - 128] = *(B + row * K + col);
      tempB[1][threadId - 128] = *(B + (row + 1) * K + col);
      tempB[2][threadId - 128] = *(B + (row + 2) * K + col);
      tempB[3][threadId - 128] = *(B + (row + 3) * K + col);
      tempB[4][threadId - 128] = *(B + (row + 4) * K + col);
      tempB[5][threadId - 128] = *(B + (row + 5) * K + col);
      tempB[6][threadId - 128] = *(B + (row + 6) * K + col);
      tempB[7][threadId - 128] = *(B + (row + 7) * K + col);
    }
    __syncthreads();

// perform wrap level computing
#pragma unroll
    for (int i = 0; i < 8; i++) {
      A1.x = tempA[wrapRow32 + threadRow4][i];
      A1.y = tempA[wrapRow32 + threadRow4 + 1][i];
      A1.z = tempA[wrapRow32 + threadRow4 + 2][i];
      A1.w = tempA[wrapRow32 + threadRow4 + 3][i];

      A1.x = tempA[wrapRow32 + threadRow4 + 4][i];
      A2.y = tempA[wrapRow32 + threadRow4 + 5][i];
      A2.z = tempA[wrapRow32 + threadRow4 + 6][i];
      A2.w = tempA[wrapRow32 + threadRow4 + 7][i];

      B1.x = tempB[i][wrapCol64 + threadCol4];
      B1.y = tempB[i][wrapCol64 + threadCol4 + 1];
      B1.z = tempB[i][wrapCol64 + threadCol4 + 2];
      B1.w = tempB[i][wrapCol64 + threadCol4 + 3];

      B2.x = tempB[i][wrapCol64 + threadCol4 + 8];
      B2.y = tempB[i][wrapCol64 + threadCol4 + 9];
      B2.z = tempB[i][wrapCol64 + threadCol4 + 10];
      B2.w = tempB[i][wrapCol64 + threadCol4 + 11];

      // fisrt 4x4
      threadResults[0][0] += A1.x * B1.x;
      threadResults[0][1] += A1.x * B1.y;
      threadResults[0][2] += A1.x * B1.z;
      threadResults[0][3] += A1.x * B1.w;
      threadResults[1][0] += A1.y * B1.x;
      threadResults[1][1] += A1.y * B1.y;
      threadResults[1][2] += A1.y * B1.z;
      threadResults[1][3] += A1.y * B1.w;
      threadResults[2][0] += A1.z * B1.x;
      threadResults[2][1] += A1.z * B1.y;
      threadResults[2][2] += A1.z * B1.z;
      threadResults[2][3] += A1.z * B1.w;
      threadResults[3][0] += A1.w * B1.x;
      threadResults[3][1] += A1.w * B1.y;
      threadResults[3][2] += A1.w * B1.z;
      threadResults[3][3] += A1.w * B1.w;

      // second 4x4
      threadResults[0][4] += A1.x * B2.x;
      threadResults[0][5] += A1.x * B2.y;
      threadResults[0][6] += A1.x * B2.z;
      threadResults[0][7] += A1.x * B2.w;
      threadResults[1][4] += A1.y * B2.x;
      threadResults[1][5] += A1.y * B2.y;
      threadResults[1][6] += A1.y * B2.z;
      threadResults[1][7] += A1.y * B2.w;
      threadResults[2][4] += A1.z * B2.x;
      threadResults[2][5] += A1.z * B2.y;
      threadResults[2][6] += A1.z * B2.z;
      threadResults[2][7] += A1.z * B2.w;
      threadResults[3][4] += A1.w * B2.x;
      threadResults[3][5] += A1.w * B2.y;
      threadResults[3][6] += A1.w * B2.z;
      threadResults[3][7] += A1.w * B2.w;

      // third 4x4
      threadResults[4][0] += A2.x * B1.x;
      threadResults[4][1] += A2.x * B1.y;
      threadResults[4][2] += A2.x * B1.z;
      threadResults[4][3] += A2.x * B1.w;
      threadResults[5][0] += A2.y * B1.x;
      threadResults[5][1] += A2.y * B1.y;
      threadResults[5][2] += A2.y * B1.z;
      threadResults[5][3] += A2.y * B1.w;
      threadResults[6][0] += A2.z * B1.x;
      threadResults[6][1] += A2.z * B1.y;
      threadResults[6][2] += A2.z * B1.z;
      threadResults[6][3] += A2.z * B1.w;
      threadResults[7][0] += A2.w * B1.x;
      threadResults[7][1] += A2.w * B1.y;
      threadResults[7][2] += A2.w * B1.z;
      threadResults[7][3] += A2.w * B1.w;

      // last 4x4
      threadResults[4][4] += A2.x * B2.x;
      threadResults[4][5] += A2.x * B2.y;
      threadResults[4][6] += A2.x * B2.z;
      threadResults[4][7] += A2.x * B2.w;
      threadResults[5][4] += A2.y * B2.x;
      threadResults[5][5] += A2.y * B2.y;
      threadResults[5][6] += A2.y * B2.z;
      threadResults[5][7] += A2.y * B2.w;
      threadResults[6][4] += A2.z * B2.x;
      threadResults[6][5] += A2.z * B2.y;
      threadResults[6][6] += A2.z * B2.z;
      threadResults[6][7] += A2.z * B2.w;
      threadResults[7][4] += A2.w * B2.x;
      threadResults[7][5] += A2.w * B2.y;
      threadResults[7][6] += A2.w * B2.z;
      threadResults[7][7] += A2.w * B2.w;
    }
  }

  // write temp thread level results back to global memory, each thread wirtes
  // 8x8 elements
  int globalRow = blockIdx.x * 128 + wrapRow32 + threadRow4;
  int globalCol = blockIdx.y * 128 + wrapCol64 + threadCol4;

  // first row
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])));
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])) + 1);
  // second row
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])) + 1);
  // third row
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])) + 1);
  // fourth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])) + 1);
  // fiveth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])) + 1);
  // sixth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])) + 1);
  // seventh row
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])) + 1);
  // last row
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])) + 1);
}

// use software prefetching to hide computation latency
// use another memory loading scheming
// DRAM Frequency         cycle/nsecond             6.79
// SM Frequency           cycle/nsecond             1.36
// Elapsed Cycles         cycle                     43,527,883
// Memory [%]             %                         48.13
// SOL DRAM               %                         5.30
// Duration               msecond                   31.89
// SOL L1/TEX Cache       %                         96.25
// SOL L2 Cache           %                         7.44
// SM Active Cycles       cycle                     41,918,687.32
// SM [%]                  %                        36.42
__global__ void sgemmTillingV4(float *A, float *B, float *C, int M, int N,
                               int K) {
  __shared__ float tempA[128][8];
  __shared__ float tempB[8][128];

  int rowA = blockIdx.x;
  int colB = blockIdx.y;
  int threadId = threadIdx.x;

  int wrapId = threadId >> 5;
  int laneId = threadId & 31;
  // map wrapId to wrapResults row and column
  int wrapRow32 = (wrapId / 2) << 5;
  int wrapCol64 = (wrapId % 2) << 6;
  // map laneId to thread level resuls row and column
  int threadRow4 = (laneId / 8) << 2;
  int threadCol4 = (laneId % 8) << 2;

  // registers to save temp results
  float threadResults[8][8] = {0};
  float4 A1[2], A2[2], B1[2], B2[2];
  float4 prefetchA1, prefetchA2, prefetchB1, prefetchB2;
  // read first tile of A and B to shared memory
  // read 128x8 data into tempA
  int row, col;
  if (threadId < 128) {
    row = (rowA * 128 + threadId);
    float *tempBasePtr = &(tempA[threadId][0]);
    *(reinterpret_cast<float4 *>(tempBasePtr)) =
        *(reinterpret_cast<float4 *>(A + row * N));

    *(reinterpret_cast<float4 *>(tempBasePtr + 4)) =
        *(reinterpret_cast<float4 *>(A + row * N + 4));
  } else  // read 128x8 data into tempB
  {
    col = colB * 128 + threadId - 128;
    tempB[0][threadId - 128] = *(B + col);
    tempB[1][threadId - 128] = *(B + 1 * K + col);
    tempB[2][threadId - 128] = *(B + 2 * K + col);
    tempB[3][threadId - 128] = *(B + 3 * K + col);
    tempB[4][threadId - 128] = *(B + 4 * K + col);
    tempB[5][threadId - 128] = *(B + 5 * K + col);
    tempB[6][threadId - 128] = *(B + 6 * K + col);
    tempB[7][threadId - 128] = *(B + 7 * K + col);
  }
  __syncthreads();

  int iter = N / 8;

  for (int i = 0; i < iter; i++) {
    // prefetch next tile of A and B into registers, after the global memory
    // access call sent, the rountine will jump to wrap based computing
    // directly, thus overlap the computation latency

    if (i < iter - 1) {
      if (threadId < 128) {
        row = (rowA * 128 + threadId);
        col = (i + 1) * 8;
        prefetchA1 = *(reinterpret_cast<float4 *>(A + row * N + col));
        prefetchA2 = *(reinterpret_cast<float4 *>(A + row * N + col + 4));
      } else  // read 128x8 data into tempB
      {
        row = (i + 1) * 8;
        col = colB * 128 + threadId - 128;
        prefetchB1.x = *(B + row * K + col);
        prefetchB1.y = *(B + (row + 1) * K + col);
        prefetchB1.z = *(B + (row + 2) * K + col);
        prefetchB1.w = *(B + (row + 3) * K + col);
        prefetchB2.x = *(B + (row + 4) * K + col);
        prefetchB2.y = *(B + (row + 5) * K + col);
        prefetchB2.z = *(B + (row + 6) * K + col);
        prefetchB2.w = *(B + (row + 7) * K + col);
      }
    }

    // perform wrap level computing
    // prepare fragments of A and B
    A1[0].x = tempA[wrapRow32 + threadRow4][0];
    A1[0].y = tempA[wrapRow32 + threadRow4 + 1][0];
    A1[0].z = tempA[wrapRow32 + threadRow4 + 2][0];
    A1[0].w = tempA[wrapRow32 + threadRow4 + 3][0];
    A2[0].x = tempA[wrapRow32 + threadRow4 + 4][0];
    A2[0].y = tempA[wrapRow32 + threadRow4 + 5][0];
    A2[0].z = tempA[wrapRow32 + threadRow4 + 6][0];
    A2[0].w = tempA[wrapRow32 + threadRow4 + 7][0];
    B1[0].x = tempB[0][wrapCol64 + threadCol4];
    B1[0].y = tempB[0][wrapCol64 + threadCol4 + 1];
    B1[0].z = tempB[0][wrapCol64 + threadCol4 + 2];
    B1[0].w = tempB[0][wrapCol64 + threadCol4 + 3];
    B2[0].x = tempB[0][wrapCol64 + threadCol4 + 8];
    B2[0].y = tempB[0][wrapCol64 + threadCol4 + 9];
    B2[0].z = tempB[0][wrapCol64 + threadCol4 + 10];
    B2[0].w = tempB[0][wrapCol64 + threadCol4 + 11];

#pragma unroll
    for (int j = 0; j < 8; j++) {
      // prefetch next fragments of A and B from shared memory to registers
      if (j < 7) {
        A1[(j + 1) & 1].x = tempA[wrapRow32 + threadRow4][j + 1];
        A1[(j + 1) & 1].y = tempA[wrapRow32 + threadRow4 + 1][j + 1];
        A1[(j + 1) & 1].z = tempA[wrapRow32 + threadRow4 + 2][j + 1];
        A1[(j + 1) & 1].w = tempA[wrapRow32 + threadRow4 + 3][j + 1];

        A2[(j + 1) & 1].x = tempA[wrapRow32 + threadRow4 + 4][j + 1];
        A2[(j + 1) & 1].y = tempA[wrapRow32 + threadRow4 + 5][j + 1];
        A2[(j + 1) & 1].z = tempA[wrapRow32 + threadRow4 + 6][j + 1];
        A2[(j + 1) & 1].w = tempA[wrapRow32 + threadRow4 + 7][j + 1];

        B1[(j + 1) & 1].x = tempB[j + 1][wrapCol64 + threadCol4];
        B1[(j + 1) & 1].y = tempB[j + 1][wrapCol64 + threadCol4 + 1];
        B1[(j + 1) & 1].z = tempB[j + 1][wrapCol64 + threadCol4 + 2];
        B1[(j + 1) & 1].w = tempB[j + 1][wrapCol64 + threadCol4 + 3];

        B2[(j + 1) & 1].x = tempB[j + 1][wrapCol64 + threadCol4 + 8];
        B2[(j + 1) & 1].y = tempB[j + 1][wrapCol64 + threadCol4 + 9];
        B2[(j + 1) & 1].z = tempB[j + 1][wrapCol64 + threadCol4 + 10];
        B2[(j + 1) & 1].w = tempB[j + 1][wrapCol64 + threadCol4 + 11];
        __syncthreads();
      }

      // fisrt 4x4
      threadResults[0][0] += A1[j & 1].x * B1[j & 1].x;
      threadResults[0][1] += A1[j & 1].x * B1[j & 1].y;
      threadResults[0][2] += A1[j & 1].x * B1[j & 1].z;
      threadResults[0][3] += A1[j & 1].x * B1[j & 1].w;
      threadResults[1][0] += A1[j & 1].y * B1[j & 1].x;
      threadResults[1][1] += A1[j & 1].y * B1[j & 1].y;
      threadResults[1][2] += A1[j & 1].y * B1[j & 1].z;
      threadResults[1][3] += A1[j & 1].y * B1[j & 1].w;
      threadResults[2][0] += A1[j & 1].z * B1[j & 1].x;
      threadResults[2][1] += A1[j & 1].z * B1[j & 1].y;
      threadResults[2][2] += A1[j & 1].z * B1[j & 1].z;
      threadResults[2][3] += A1[j & 1].z * B1[j & 1].w;
      threadResults[3][0] += A1[j & 1].w * B1[j & 1].x;
      threadResults[3][1] += A1[j & 1].w * B1[j & 1].y;
      threadResults[3][2] += A1[j & 1].w * B1[j & 1].z;
      threadResults[3][3] += A1[j & 1].w * B1[j & 1].w;

      // second 4x4
      threadResults[0][4] += A1[j & 1].x * B2[j & 1].x;
      threadResults[0][5] += A1[j & 1].x * B2[j & 1].y;
      threadResults[0][6] += A1[j & 1].x * B2[j & 1].z;
      threadResults[0][7] += A1[j & 1].x * B2[j & 1].w;
      threadResults[1][4] += A1[j & 1].y * B2[j & 1].x;
      threadResults[1][5] += A1[j & 1].y * B2[j & 1].y;
      threadResults[1][6] += A1[j & 1].y * B2[j & 1].z;
      threadResults[1][7] += A1[j & 1].y * B2[j & 1].w;
      threadResults[2][4] += A1[j & 1].z * B2[j & 1].x;
      threadResults[2][5] += A1[j & 1].z * B2[j & 1].y;
      threadResults[2][6] += A1[j & 1].z * B2[j & 1].z;
      threadResults[2][7] += A1[j & 1].z * B2[j & 1].w;
      threadResults[3][4] += A1[j & 1].w * B2[j & 1].x;
      threadResults[3][5] += A1[j & 1].w * B2[j & 1].y;
      threadResults[3][6] += A1[j & 1].w * B2[j & 1].z;
      threadResults[3][7] += A1[j & 1].w * B2[j & 1].w;

      // third 4x4
      threadResults[4][0] += A2[j & 1].x * B1[j & 1].x;
      threadResults[4][1] += A2[j & 1].x * B1[j & 1].y;
      threadResults[4][2] += A2[j & 1].x * B1[j & 1].z;
      threadResults[4][3] += A2[j & 1].x * B1[j & 1].w;
      threadResults[5][0] += A2[j & 1].y * B1[j & 1].x;
      threadResults[5][1] += A2[j & 1].y * B1[j & 1].y;
      threadResults[5][2] += A2[j & 1].y * B1[j & 1].z;
      threadResults[5][3] += A2[j & 1].y * B1[j & 1].w;
      threadResults[6][0] += A2[j & 1].z * B1[j & 1].x;
      threadResults[6][1] += A2[j & 1].z * B1[j & 1].y;
      threadResults[6][2] += A2[j & 1].z * B1[j & 1].z;
      threadResults[6][3] += A2[j & 1].z * B1[j & 1].w;
      threadResults[7][0] += A2[j & 1].w * B1[j & 1].x;
      threadResults[7][1] += A2[j & 1].w * B1[j & 1].y;
      threadResults[7][2] += A2[j & 1].w * B1[j & 1].z;
      threadResults[7][3] += A2[j & 1].w * B1[j & 1].w;

      // last 4x4
      threadResults[4][4] += A2[j & 1].x * B2[j & 1].x;
      threadResults[4][5] += A2[j & 1].x * B2[j & 1].y;
      threadResults[4][6] += A2[j & 1].x * B2[j & 1].z;
      threadResults[4][7] += A2[j & 1].x * B2[j & 1].w;
      threadResults[5][4] += A2[j & 1].y * B2[j & 1].x;
      threadResults[5][5] += A2[j & 1].y * B2[j & 1].y;
      threadResults[5][6] += A2[j & 1].y * B2[j & 1].z;
      threadResults[5][7] += A2[j & 1].y * B2[j & 1].w;
      threadResults[6][4] += A2[j & 1].z * B2[j & 1].x;
      threadResults[6][5] += A2[j & 1].z * B2[j & 1].y;
      threadResults[6][6] += A2[j & 1].z * B2[j & 1].z;
      threadResults[6][7] += A2[j & 1].z * B2[j & 1].w;
      threadResults[7][4] += A2[j & 1].w * B2[j & 1].x;
      threadResults[7][5] += A2[j & 1].w * B2[j & 1].y;
      threadResults[7][6] += A2[j & 1].w * B2[j & 1].z;
      threadResults[7][7] += A2[j & 1].w * B2[j & 1].w;
    }
    __syncthreads();

    // write a tile of A and B into shared memory from registers
    if (i < iter - 1) {
      if (threadId < 128) {
        float *tempBasePtr = &(tempA[threadId][0]);
        *(reinterpret_cast<float4 *>(tempBasePtr)) = prefetchA1;

        *(reinterpret_cast<float4 *>(tempBasePtr + 4)) = prefetchA2;
      } else  // read 128x8 data into tempB
      {
        tempB[0][threadId - 128] = prefetchB1.x;
        tempB[1][threadId - 128] = prefetchB1.y;
        tempB[2][threadId - 128] = prefetchB1.z;
        tempB[3][threadId - 128] = prefetchB1.w;
        tempB[4][threadId - 128] = prefetchB2.x;
        tempB[5][threadId - 128] = prefetchB2.y;
        tempB[6][threadId - 128] = prefetchB2.z;
        tempB[7][threadId - 128] = prefetchB2.w;
      }
    }
  }

  // write temp thread level results back to global memory, each thread wirtes
  // 8x8 elements
  int globalRow = (rowA << 7) + wrapRow32 + threadRow4;
  int globalCol = (colB << 7) + wrapCol64 + threadCol4;

  // first row
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])));
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])) + 1);
  // second row
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])) + 1);
  // third row
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])) + 1);
  // fourth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])) + 1);
  // fiveth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])) + 1);
  // sixth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])) + 1);
  // seventh row
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])) + 1);
  // last row
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])) + 1);
}

// use another memory loading scheming
// DRAM Frequency         cycle/nsecond             6.79
// SM Frequency           cycle/nsecond             1.36
// Elapsed Cycles         cycle                     41,097,138
// Memory [%]             %                         48.05
// SOL DRAM               %                         5.52
// Duration               msecond                   30.11
// SOL L1/TEX Cache       %                         96.10
// SOL L2 Cache           %                         7.88
// SM Active Cycles       cycle                     39,506,276.25
// SM [%]                  %                        38.65
__global__ void sgemmTillingV5(float *A, float *B, float *C, int M, int N,
                               int K) {
  __shared__ float tempA[128][8];
  __shared__ float tempB[8][128];

  int rowA = blockIdx.x;
  int colB = blockIdx.y;
  int threadId = threadIdx.x;

  int wrapId = threadId >> 5;
  int laneId = threadId & 31;
  // map wrapId to wrapResults row and column
  int wrapRow32 = (wrapId / 2) << 5;
  int wrapCol64 = (wrapId % 2) << 6;
  // map laneId to thread level resuls row and column
  int threadRow4 = (laneId / 8) << 2;
  int threadCol4 = (laneId % 8) << 2;

  // registers to save temp results
  float threadResults[8][8] = {0};
  float4 A1[2], A2[2], B1[2], B2[2];
  float4 prefetchA, prefetchB;

  // read first tile of A and B to shared memory
  // read 128x8 data into tempA
  int row_a = threadId >> 1, col_a = (threadId & 1) << 2;
  float *tempBasePtrA = &(tempA[row_a][col_a]);
  *(reinterpret_cast<float4 *>(tempBasePtrA)) =
      *(reinterpret_cast<float4 *>(A + (row_a + rowA * 128) * N + col_a));

  int row_b = threadId >> 5, col_b = (threadId & 31) << 2;
  float *tempBasePtrB = &(tempB[row_b][col_b]);
  *(reinterpret_cast<float4 *>(tempBasePtrB)) =
      *(reinterpret_cast<float4 *>(A + row_b * N + col_b + colB * 128));
  __syncthreads();

  int iter = N / 8;
  for (int i = 0; i < iter; i++) {
    // prefetch next tile of A and B into registers, after the global memory
    // access call sent, the rountine will jump to wrap based computing
    // directly, thus overlap the computation latency

    if (i < iter - 1) {
      prefetchA = *(reinterpret_cast<float4 *>(A + (row_a + rowA * 128) * N +
                                               col_a + ((i + 1) << 3)));

      prefetchB = *(reinterpret_cast<float4 *>(A + (row_b + (i + 1) * 8) * N +
                                               col_b + colB * 128));
    }

    // perform wrap level computing
    // prepare fragments of A and B
    A1[0].x = tempA[wrapRow32 + threadRow4][0];
    A1[0].y = tempA[wrapRow32 + threadRow4 + 1][0];
    A1[0].z = tempA[wrapRow32 + threadRow4 + 2][0];
    A1[0].w = tempA[wrapRow32 + threadRow4 + 3][0];
    A2[0].x = tempA[wrapRow32 + threadRow4 + 4][0];
    A2[0].y = tempA[wrapRow32 + threadRow4 + 5][0];
    A2[0].z = tempA[wrapRow32 + threadRow4 + 6][0];
    A2[0].w = tempA[wrapRow32 + threadRow4 + 7][0];
    B1[0].x = tempB[0][wrapCol64 + threadCol4];
    B1[0].y = tempB[0][wrapCol64 + threadCol4 + 1];
    B1[0].z = tempB[0][wrapCol64 + threadCol4 + 2];
    B1[0].w = tempB[0][wrapCol64 + threadCol4 + 3];
    B2[0].x = tempB[0][wrapCol64 + threadCol4 + 8];
    B2[0].y = tempB[0][wrapCol64 + threadCol4 + 9];
    B2[0].z = tempB[0][wrapCol64 + threadCol4 + 10];
    B2[0].w = tempB[0][wrapCol64 + threadCol4 + 11];

#pragma unroll
    for (int j = 0; j < 8; j++) {
      // prefetch next fragments of A and B from shared memory to registers
      if (j < 7) {
        A1[(j + 1) & 1].x = tempA[wrapRow32 + threadRow4][j + 1];
        A1[(j + 1) & 1].y = tempA[wrapRow32 + threadRow4 + 1][j + 1];
        A1[(j + 1) & 1].z = tempA[wrapRow32 + threadRow4 + 2][j + 1];
        A1[(j + 1) & 1].w = tempA[wrapRow32 + threadRow4 + 3][j + 1];

        A2[(j + 1) & 1].x = tempA[wrapRow32 + threadRow4 + 4][j + 1];
        A2[(j + 1) & 1].y = tempA[wrapRow32 + threadRow4 + 5][j + 1];
        A2[(j + 1) & 1].z = tempA[wrapRow32 + threadRow4 + 6][j + 1];
        A2[(j + 1) & 1].w = tempA[wrapRow32 + threadRow4 + 7][j + 1];

        B1[(j + 1) & 1].x = tempB[j + 1][wrapCol64 + threadCol4];
        B1[(j + 1) & 1].y = tempB[j + 1][wrapCol64 + threadCol4 + 1];
        B1[(j + 1) & 1].z = tempB[j + 1][wrapCol64 + threadCol4 + 2];
        B1[(j + 1) & 1].w = tempB[j + 1][wrapCol64 + threadCol4 + 3];

        B2[(j + 1) & 1].x = tempB[j + 1][wrapCol64 + threadCol4 + 8];
        B2[(j + 1) & 1].y = tempB[j + 1][wrapCol64 + threadCol4 + 9];
        B2[(j + 1) & 1].z = tempB[j + 1][wrapCol64 + threadCol4 + 10];
        B2[(j + 1) & 1].w = tempB[j + 1][wrapCol64 + threadCol4 + 11];
        __syncthreads();
      }

      // fisrt 4x4
      threadResults[0][0] += A1[j & 1].x * B1[j & 1].x;
      threadResults[0][1] += A1[j & 1].x * B1[j & 1].y;
      threadResults[0][2] += A1[j & 1].x * B1[j & 1].z;
      threadResults[0][3] += A1[j & 1].x * B1[j & 1].w;
      threadResults[1][0] += A1[j & 1].y * B1[j & 1].x;
      threadResults[1][1] += A1[j & 1].y * B1[j & 1].y;
      threadResults[1][2] += A1[j & 1].y * B1[j & 1].z;
      threadResults[1][3] += A1[j & 1].y * B1[j & 1].w;
      threadResults[2][0] += A1[j & 1].z * B1[j & 1].x;
      threadResults[2][1] += A1[j & 1].z * B1[j & 1].y;
      threadResults[2][2] += A1[j & 1].z * B1[j & 1].z;
      threadResults[2][3] += A1[j & 1].z * B1[j & 1].w;
      threadResults[3][0] += A1[j & 1].w * B1[j & 1].x;
      threadResults[3][1] += A1[j & 1].w * B1[j & 1].y;
      threadResults[3][2] += A1[j & 1].w * B1[j & 1].z;
      threadResults[3][3] += A1[j & 1].w * B1[j & 1].w;

      // second 4x4
      threadResults[0][4] += A1[j & 1].x * B2[j & 1].x;
      threadResults[0][5] += A1[j & 1].x * B2[j & 1].y;
      threadResults[0][6] += A1[j & 1].x * B2[j & 1].z;
      threadResults[0][7] += A1[j & 1].x * B2[j & 1].w;
      threadResults[1][4] += A1[j & 1].y * B2[j & 1].x;
      threadResults[1][5] += A1[j & 1].y * B2[j & 1].y;
      threadResults[1][6] += A1[j & 1].y * B2[j & 1].z;
      threadResults[1][7] += A1[j & 1].y * B2[j & 1].w;
      threadResults[2][4] += A1[j & 1].z * B2[j & 1].x;
      threadResults[2][5] += A1[j & 1].z * B2[j & 1].y;
      threadResults[2][6] += A1[j & 1].z * B2[j & 1].z;
      threadResults[2][7] += A1[j & 1].z * B2[j & 1].w;
      threadResults[3][4] += A1[j & 1].w * B2[j & 1].x;
      threadResults[3][5] += A1[j & 1].w * B2[j & 1].y;
      threadResults[3][6] += A1[j & 1].w * B2[j & 1].z;
      threadResults[3][7] += A1[j & 1].w * B2[j & 1].w;

      // third 4x4
      threadResults[4][0] += A2[j & 1].x * B1[j & 1].x;
      threadResults[4][1] += A2[j & 1].x * B1[j & 1].y;
      threadResults[4][2] += A2[j & 1].x * B1[j & 1].z;
      threadResults[4][3] += A2[j & 1].x * B1[j & 1].w;
      threadResults[5][0] += A2[j & 1].y * B1[j & 1].x;
      threadResults[5][1] += A2[j & 1].y * B1[j & 1].y;
      threadResults[5][2] += A2[j & 1].y * B1[j & 1].z;
      threadResults[5][3] += A2[j & 1].y * B1[j & 1].w;
      threadResults[6][0] += A2[j & 1].z * B1[j & 1].x;
      threadResults[6][1] += A2[j & 1].z * B1[j & 1].y;
      threadResults[6][2] += A2[j & 1].z * B1[j & 1].z;
      threadResults[6][3] += A2[j & 1].z * B1[j & 1].w;
      threadResults[7][0] += A2[j & 1].w * B1[j & 1].x;
      threadResults[7][1] += A2[j & 1].w * B1[j & 1].y;
      threadResults[7][2] += A2[j & 1].w * B1[j & 1].z;
      threadResults[7][3] += A2[j & 1].w * B1[j & 1].w;

      // last 4x4
      threadResults[4][4] += A2[j & 1].x * B2[j & 1].x;
      threadResults[4][5] += A2[j & 1].x * B2[j & 1].y;
      threadResults[4][6] += A2[j & 1].x * B2[j & 1].z;
      threadResults[4][7] += A2[j & 1].x * B2[j & 1].w;
      threadResults[5][4] += A2[j & 1].y * B2[j & 1].x;
      threadResults[5][5] += A2[j & 1].y * B2[j & 1].y;
      threadResults[5][6] += A2[j & 1].y * B2[j & 1].z;
      threadResults[5][7] += A2[j & 1].y * B2[j & 1].w;
      threadResults[6][4] += A2[j & 1].z * B2[j & 1].x;
      threadResults[6][5] += A2[j & 1].z * B2[j & 1].y;
      threadResults[6][6] += A2[j & 1].z * B2[j & 1].z;
      threadResults[6][7] += A2[j & 1].z * B2[j & 1].w;
      threadResults[7][4] += A2[j & 1].w * B2[j & 1].x;
      threadResults[7][5] += A2[j & 1].w * B2[j & 1].y;
      threadResults[7][6] += A2[j & 1].w * B2[j & 1].z;
      threadResults[7][7] += A2[j & 1].w * B2[j & 1].w;
    }
    __syncthreads();

    // write a tile of A and B into shared memory from registers
    if (i < iter - 1) {
      tempBasePtrA = &(tempA[row_a][col_a]);
      *(reinterpret_cast<float4 *>(tempBasePtrA)) = prefetchA;

      tempBasePtrB = &(tempB[row_b][col_b]);
      *(reinterpret_cast<float4 *>(tempBasePtrB)) = prefetchB;
    }
  }

  // write temp thread level results back to global memory, each thread wirtes
  // 8x8 elements
  int globalRow = (rowA << 7) + wrapRow32 + threadRow4;
  int globalCol = (colB << 7) + wrapCol64 + threadCol4;

  // first row
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])));
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])) + 1);
  // second row
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])) + 1);
  // third row
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])) + 1);
  // fourth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])) + 1);
  // fiveth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])) + 1);
  // sixth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])) + 1);
  // seventh row
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])) + 1);
  // last row
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])) + 1);
}

// reorder memory access to tile of A and B in shared memory, use column major
__global__ void sgemmTillingV6(float *A, float *B, float *C, int M, int N,
                               int K) {
  __shared__ float tempA[8][128];
  __shared__ float tempB[128][8];

  int rowA = blockIdx.x;
  int colB = blockIdx.y;
  int threadId = threadIdx.x;

  int wrapId = threadId >> 5;
  int laneId = threadId & 31;
  // map wrapId to wrapResults row and column
  int wrapRow32 = (wrapId / 2) << 5;
  int wrapCol64 = (wrapId % 2) << 6;
  // map laneId to thread level resuls row and column
  int threadRow4 = (laneId / 8) << 2;
  int threadCol4 = (laneId % 8) << 2;

  // registers to save temp results
  float threadResults[8][8] = {0};
  float4 A1[2], A2[2], B1[2], B2[2];
  float4 prefetchA, prefetchB;

  // read first tile of A and B to shared memory
  // read 128x8 data into tempA
  int row_a = threadId >> 1, col_a = (threadId & 1) << 2;
  float *tempBasePtrA = &(tempA[row_a][col_a]);
  *(reinterpret_cast<float4 *>(tempBasePtrA)) =
      *(reinterpret_cast<float4 *>(A + (row_a + rowA * 128) * N + col_a));

  int row_b = threadId >> 5, col_b = (threadId & 31) << 2;
  float *tempBasePtrB = &(tempB[row_b][col_b]);
  *(reinterpret_cast<float4 *>(tempBasePtrB)) =
      *(reinterpret_cast<float4 *>(A + row_b * N + col_b + colB * 128));
  __syncthreads();

  int iter = N / 8;
  for (int i = 0; i < iter; i++) {
    // prefetch next tile of A and B into registers, after the global memory
    // access call sent, the rountine will jump to wrap based computing
    // directly, thus overlap the computation latency

    if (i < iter - 1) {
      prefetchA = *(reinterpret_cast<float4 *>(A + (row_a + rowA * 128) * N +
                                               col_a + ((i + 1) << 3)));

      prefetchB = *(reinterpret_cast<float4 *>(A + (row_b + (i + 1) * 8) * N +
                                               col_b + colB * 128));
    }

    // perform wrap level computing
    // prepare fragments of A and B
    A1[0].x = tempA[wrapRow32 + threadRow4][0];
    A1[0].y = tempA[wrapRow32 + threadRow4 + 1][0];
    A1[0].z = tempA[wrapRow32 + threadRow4 + 2][0];
    A1[0].w = tempA[wrapRow32 + threadRow4 + 3][0];
    A2[0].x = tempA[wrapRow32 + threadRow4 + 4][0];
    A2[0].y = tempA[wrapRow32 + threadRow4 + 5][0];
    A2[0].z = tempA[wrapRow32 + threadRow4 + 6][0];
    A2[0].w = tempA[wrapRow32 + threadRow4 + 7][0];
    B1[0].x = tempB[0][wrapCol64 + threadCol4];
    B1[0].y = tempB[0][wrapCol64 + threadCol4 + 1];
    B1[0].z = tempB[0][wrapCol64 + threadCol4 + 2];
    B1[0].w = tempB[0][wrapCol64 + threadCol4 + 3];
    B2[0].x = tempB[0][wrapCol64 + threadCol4 + 8];
    B2[0].y = tempB[0][wrapCol64 + threadCol4 + 9];
    B2[0].z = tempB[0][wrapCol64 + threadCol4 + 10];
    B2[0].w = tempB[0][wrapCol64 + threadCol4 + 11];

#pragma unroll
    for (int j = 0; j < 8; j++) {
      // prefetch next fragments of A and B from shared memory to registers
      if (j < 7) {
        A1[(j + 1) & 1].x = tempA[wrapRow32 + threadRow4][j + 1];
        A1[(j + 1) & 1].y = tempA[wrapRow32 + threadRow4 + 1][j + 1];
        A1[(j + 1) & 1].z = tempA[wrapRow32 + threadRow4 + 2][j + 1];
        A1[(j + 1) & 1].w = tempA[wrapRow32 + threadRow4 + 3][j + 1];

        A2[(j + 1) & 1].x = tempA[wrapRow32 + threadRow4 + 4][j + 1];
        A2[(j + 1) & 1].y = tempA[wrapRow32 + threadRow4 + 5][j + 1];
        A2[(j + 1) & 1].z = tempA[wrapRow32 + threadRow4 + 6][j + 1];
        A2[(j + 1) & 1].w = tempA[wrapRow32 + threadRow4 + 7][j + 1];

        B1[(j + 1) & 1].x = tempB[j + 1][wrapCol64 + threadCol4];
        B1[(j + 1) & 1].y = tempB[j + 1][wrapCol64 + threadCol4 + 1];
        B1[(j + 1) & 1].z = tempB[j + 1][wrapCol64 + threadCol4 + 2];
        B1[(j + 1) & 1].w = tempB[j + 1][wrapCol64 + threadCol4 + 3];

        B2[(j + 1) & 1].x = tempB[j + 1][wrapCol64 + threadCol4 + 8];
        B2[(j + 1) & 1].y = tempB[j + 1][wrapCol64 + threadCol4 + 9];
        B2[(j + 1) & 1].z = tempB[j + 1][wrapCol64 + threadCol4 + 10];
        B2[(j + 1) & 1].w = tempB[j + 1][wrapCol64 + threadCol4 + 11];
        __syncthreads();
      }

      // fisrt 4x4
      threadResults[0][0] += A1[j & 1].x * B1[j & 1].x;
      threadResults[0][1] += A1[j & 1].x * B1[j & 1].y;
      threadResults[0][2] += A1[j & 1].x * B1[j & 1].z;
      threadResults[0][3] += A1[j & 1].x * B1[j & 1].w;
      threadResults[1][0] += A1[j & 1].y * B1[j & 1].x;
      threadResults[1][1] += A1[j & 1].y * B1[j & 1].y;
      threadResults[1][2] += A1[j & 1].y * B1[j & 1].z;
      threadResults[1][3] += A1[j & 1].y * B1[j & 1].w;
      threadResults[2][0] += A1[j & 1].z * B1[j & 1].x;
      threadResults[2][1] += A1[j & 1].z * B1[j & 1].y;
      threadResults[2][2] += A1[j & 1].z * B1[j & 1].z;
      threadResults[2][3] += A1[j & 1].z * B1[j & 1].w;
      threadResults[3][0] += A1[j & 1].w * B1[j & 1].x;
      threadResults[3][1] += A1[j & 1].w * B1[j & 1].y;
      threadResults[3][2] += A1[j & 1].w * B1[j & 1].z;
      threadResults[3][3] += A1[j & 1].w * B1[j & 1].w;

      // second 4x4
      threadResults[0][4] += A1[j & 1].x * B2[j & 1].x;
      threadResults[0][5] += A1[j & 1].x * B2[j & 1].y;
      threadResults[0][6] += A1[j & 1].x * B2[j & 1].z;
      threadResults[0][7] += A1[j & 1].x * B2[j & 1].w;
      threadResults[1][4] += A1[j & 1].y * B2[j & 1].x;
      threadResults[1][5] += A1[j & 1].y * B2[j & 1].y;
      threadResults[1][6] += A1[j & 1].y * B2[j & 1].z;
      threadResults[1][7] += A1[j & 1].y * B2[j & 1].w;
      threadResults[2][4] += A1[j & 1].z * B2[j & 1].x;
      threadResults[2][5] += A1[j & 1].z * B2[j & 1].y;
      threadResults[2][6] += A1[j & 1].z * B2[j & 1].z;
      threadResults[2][7] += A1[j & 1].z * B2[j & 1].w;
      threadResults[3][4] += A1[j & 1].w * B2[j & 1].x;
      threadResults[3][5] += A1[j & 1].w * B2[j & 1].y;
      threadResults[3][6] += A1[j & 1].w * B2[j & 1].z;
      threadResults[3][7] += A1[j & 1].w * B2[j & 1].w;

      // third 4x4
      threadResults[4][0] += A2[j & 1].x * B1[j & 1].x;
      threadResults[4][1] += A2[j & 1].x * B1[j & 1].y;
      threadResults[4][2] += A2[j & 1].x * B1[j & 1].z;
      threadResults[4][3] += A2[j & 1].x * B1[j & 1].w;
      threadResults[5][0] += A2[j & 1].y * B1[j & 1].x;
      threadResults[5][1] += A2[j & 1].y * B1[j & 1].y;
      threadResults[5][2] += A2[j & 1].y * B1[j & 1].z;
      threadResults[5][3] += A2[j & 1].y * B1[j & 1].w;
      threadResults[6][0] += A2[j & 1].z * B1[j & 1].x;
      threadResults[6][1] += A2[j & 1].z * B1[j & 1].y;
      threadResults[6][2] += A2[j & 1].z * B1[j & 1].z;
      threadResults[6][3] += A2[j & 1].z * B1[j & 1].w;
      threadResults[7][0] += A2[j & 1].w * B1[j & 1].x;
      threadResults[7][1] += A2[j & 1].w * B1[j & 1].y;
      threadResults[7][2] += A2[j & 1].w * B1[j & 1].z;
      threadResults[7][3] += A2[j & 1].w * B1[j & 1].w;

      // last 4x4
      threadResults[4][4] += A2[j & 1].x * B2[j & 1].x;
      threadResults[4][5] += A2[j & 1].x * B2[j & 1].y;
      threadResults[4][6] += A2[j & 1].x * B2[j & 1].z;
      threadResults[4][7] += A2[j & 1].x * B2[j & 1].w;
      threadResults[5][4] += A2[j & 1].y * B2[j & 1].x;
      threadResults[5][5] += A2[j & 1].y * B2[j & 1].y;
      threadResults[5][6] += A2[j & 1].y * B2[j & 1].z;
      threadResults[5][7] += A2[j & 1].y * B2[j & 1].w;
      threadResults[6][4] += A2[j & 1].z * B2[j & 1].x;
      threadResults[6][5] += A2[j & 1].z * B2[j & 1].y;
      threadResults[6][6] += A2[j & 1].z * B2[j & 1].z;
      threadResults[6][7] += A2[j & 1].z * B2[j & 1].w;
      threadResults[7][4] += A2[j & 1].w * B2[j & 1].x;
      threadResults[7][5] += A2[j & 1].w * B2[j & 1].y;
      threadResults[7][6] += A2[j & 1].w * B2[j & 1].z;
      threadResults[7][7] += A2[j & 1].w * B2[j & 1].w;
    }
    __syncthreads();

    // write a tile of A and B into shared memory from registers
    if (i < iter - 1) {
      tempBasePtrA = &(tempA[row_a][col_a]);
      *(reinterpret_cast<float4 *>(tempBasePtrA)) = prefetchA;

      tempBasePtrB = &(tempB[row_b][col_b]);
      *(reinterpret_cast<float4 *>(tempBasePtrB)) = prefetchB;
    }
  }

  // write temp thread level results back to global memory, each thread wirtes
  // 8x8 elements
  int globalRow = (rowA << 7) + wrapRow32 + threadRow4;
  int globalCol = (colB << 7) + wrapCol64 + threadCol4;

  // first row
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])));
  *(reinterpret_cast<float4 *>(C + globalRow * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[0][0])) + 1);
  // second row
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 1) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[1][0])) + 1);
  // third row
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 2) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[2][0])) + 1);
  // fourth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 3) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[3][0])) + 1);
  // fiveth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 4) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[4][0])) + 1);
  // sixth row
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 5) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[5][0])) + 1);
  // seventh row
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 6) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[6][0])) + 1);
  // last row
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol)) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])));
  *(reinterpret_cast<float4 *>(C + (globalRow + 7) * K + globalCol) + 8) =
      *(reinterpret_cast<float4 *>(&(threadResults[7][0])) + 1);
}

void testSgemmNaive() {
  int M = 4096, N = 4096, K = 4096;
  std::vector<float> A(M * N, 0);
  std::vector<float> B(N * K, 0);
  std::vector<float> C(M * K, 0);

  setValue(A, M * N);
  setValue(B, N * K);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float) * M * N);
  cudaMalloc(&d_B, sizeof(float) * N * K);
  cudaMalloc(&d_C, sizeof(float) * M * K);
  cudaMemcpy(d_A, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);

  dim3 block(128, 128);
  dim3 grid((M + block.x - 1) / block.x, (K + block.y - 1) / block.y);

  sgemmNaive<<<1, 1>>>(d_A, d_B, d_C, M, N, K);

  cudaMemcpy(C.data(), d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

  std::cout << "results: " << C[0] << " " << C[1] << " " << C[2] << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void testSgemmTilling() {
  int M = 4096, N = 4096, K = 4096;
  std::vector<float> A(M * N, 0);
  std::vector<float> B(N * K, 0);
  std::vector<float> C(M * K, 0);

  setValue(A, M * N);
  setValue(B, N * K);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float) * M * N);
  cudaMalloc(&d_B, sizeof(float) * N * K);
  cudaMalloc(&d_C, sizeof(float) * M * K);
  cudaMemcpy(d_A, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (K + block.y - 1) / block.y);

  (sgemmTilling<16, 16>)<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  cudaMemcpy(C.data(), d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void testSgemmTillingV2() {
  int M = 4096, N = 4096, K = 4096;
  std::vector<float> A(M * N, 0);
  std::vector<float> B(N * K, 0);
  std::vector<float> C(M * K, 0);

  setValue(A, M * N);
  setValue(B, N * K);

  std::cout << "A: " << A[0] << " " << A[1] << " " << A[2] << " " << A[3]
            << std::endl;

  std::cout << "B: " << B[0] << " " << B[1] << " " << B[2] << " " << B[3]
            << std::endl;

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float) * M * N);
  cudaMalloc(&d_B, sizeof(float) * N * K);
  cudaMalloc(&d_C, sizeof(float) * M * K);
  cudaMemcpy(d_A, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((M + 128 - 1) / 128, (K + 128 - 1) / 128);

  sgemmTillingV2<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  CHECK_ERROR(cudaGetLastError());

  cudaMemcpy(C.data(), d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

  std::cout << "result: " << C[1] << " " << C[2] << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void testSgemmTillingV3() {
  int M = 4096, N = 4096, K = 4096;
  std::vector<float> A(M * N, 0);
  std::vector<float> B(N * K, 0);
  std::vector<float> C(M * K, 0);

  setValue(A, M * N);
  setValue(B, N * K);

  std::cout << "A: " << A[0] << " " << A[1] << " " << A[2] << " " << A[3]
            << std::endl;

  std::cout << "B: " << B[0] << " " << B[1] << " " << B[2] << " " << B[3]
            << std::endl;

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float) * M * N);
  cudaMalloc(&d_B, sizeof(float) * N * K);
  cudaMalloc(&d_C, sizeof(float) * M * K);
  cudaMemcpy(d_A, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);

  dim3 block(256, 1);
  dim3 grid((M + 128 - 1) / 128, (K + 128 - 1) / 128);

  sgemmTillingV3<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  CHECK_ERROR(cudaGetLastError());

  cudaMemcpy(C.data(), d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

  std::cout << "result: " << C[1] << " " << C[2] << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void testSgemmTillingV3_5() {
  int M = 4096, N = 4096, K = 4096;
  std::vector<float> A(M * N, 0);
  std::vector<float> B(N * K, 0);
  std::vector<float> C(M * K, 0);

  setValue(A, M * N);
  setValue(B, N * K);

  std::cout << "A: " << A[0] << " " << A[1] << " " << A[2] << " " << A[3]
            << std::endl;

  std::cout << "B: " << B[0] << " " << B[1] << " " << B[2] << " " << B[3]
            << std::endl;

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float) * M * N);
  cudaMalloc(&d_B, sizeof(float) * N * K);
  cudaMalloc(&d_C, sizeof(float) * M * K);
  cudaMemcpy(d_A, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);

  dim3 block(256, 1);
  dim3 grid((M + 128 - 1) / 128, (K + 128 - 1) / 128);

  sgemmTillingV3_5<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  CHECK_ERROR(cudaGetLastError());

  cudaMemcpy(C.data(), d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

  std::cout << "result: " << C[1] << " " << C[2] << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void testSgemmTillingV4() {
  int M = 4096, N = 4096, K = 4096;
  std::vector<float> A(M * N, 0);
  std::vector<float> B(N * K, 0);
  std::vector<float> C(M * K, 0);

  setValue(A, M * N);
  setValue(B, N * K);

  std::cout << "A: " << A[0] << " " << A[1] << " " << A[2] << " " << A[3]
            << std::endl;

  std::cout << "B: " << B[0] << " " << B[1] << " " << B[2] << " " << B[3]
            << std::endl;

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float) * M * N);
  cudaMalloc(&d_B, sizeof(float) * N * K);
  cudaMalloc(&d_C, sizeof(float) * M * K);
  cudaMemcpy(d_A, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);

  dim3 block(256, 1);
  dim3 grid((M + 128 - 1) / 128, (K + 128 - 1) / 128);

  sgemmTillingV4<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  CHECK_ERROR(cudaGetLastError());

  cudaMemcpy(C.data(), d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

  std::cout << "result: " << C[1] << " " << C[2] << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void testSgemmTillingV5() {
  int M = 4096, N = 4096, K = 4096;
  std::vector<float> A(M * N, 0);
  std::vector<float> B(N * K, 0);
  std::vector<float> C(M * K, 0);

  setValue(A, M * N);
  setValue(B, N * K);

  std::cout << "A: " << A[0] << " " << A[1] << " " << A[2] << " " << A[3]
            << std::endl;

  std::cout << "B: " << B[0] << " " << B[1] << " " << B[2] << " " << B[3]
            << std::endl;

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float) * M * N);
  cudaMalloc(&d_B, sizeof(float) * N * K);
  cudaMalloc(&d_C, sizeof(float) * M * K);
  cudaMemcpy(d_A, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);

  dim3 block(256, 1);
  dim3 grid((M + 128 - 1) / 128, (K + 128 - 1) / 128);

  sgemmTillingV5<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  CHECK_ERROR(cudaGetLastError());

  cudaMemcpy(C.data(), d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

  std::cout << "result: " << C[1] << " " << C[2] << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {
  // testSgemmNaive();
  // testSgemmTilling();

  // testSgemmTillingV2();
  // testSgemmTillingV3();
  // testSgemmTillingV3_5();
  testSgemmTillingV4();
  // testSgemmTillingV5();
}