#include "../common/book.h"
#include "lock.h"

#define imin(a,b) (a<b?a:b)
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 33*1024*1024; // 33MB
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock); // max 32 blocks

// lock: govern access to output buffer <c>
__global__ void dot(Lock lock, float *a, float *b, float *c) {
  __shared__ float cache[threadsPerBlock]; // (256,); for each block
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIdx = threadIdx.x;

  float temp = 0;
  while(tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheIdx] = temp;
  __syncthreads();

  // reduction - only cache[0] is meaning ful
  int i = blockDim.x/2;
  while(i!=0) {
    if (cacheIdx < i) {
      cache[cacheIdx] += cache[cacheIdx+i];
    }
    __syncthreads();
    i /= 2;
  }
  if (cacheIdx == 0) {
    lock.lock();
    *c += cache[0];
    lock.unlock();
  }
}

int main() {
  float *a, *b, c = 0;
  float *dev_a, *dev_b, *dev_c;

  // allocate memory on the CPU
  a = (float*)malloc(N*sizeof(float));
  b = (float*)malloc(N*sizeof(float));
  // allocate memory on the GPU
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(float))); // allocate ONE float on GPU
  // fill <a> and <b> on CPU
  for (int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i*2;
  }
  // copy <a>,<b>,<c> to GPU
  HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_c, &c, sizeof(float), cudaMemcpyHostToDevice));
  // lock and execute kernel
  Lock lock;
  dot<<<blocksPerGrid,threadsPerBlock>>>(lock, dev_a, dev_b, dev_c);
  // <c> back to CPU
  HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost));
  
  // check
  printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float)(N-1)));
  // clean
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  free(a);
  free(b);
}