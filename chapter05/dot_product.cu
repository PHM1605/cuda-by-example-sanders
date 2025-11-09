#include "../common/book.h"

#define imin(a, b) (a<b?a:b)
// 2*(1**2 + 2**2 + 3**2 + ...) = 2*sum_squares(N-1) with x=N-1
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 33*124;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock); // minimum 32 blocks per grid; 2nd term means (ceiling of N/threadsPerBlock)

__global__ void dot(float *a, float *b, float *c) {
  __shared__ float cache[threadsPerBlock]; // ONE cache[] for EACH BLOCK
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  while(tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  // set  the cache value of THAT thread in block
  cache[cacheIndex] = temp;

  // synchronize threads in this block
  __syncthreads();

  // split threads in half, add pair by pair
  int i = blockDim.x/2;
  while (i!=0) {
    // only threads in the first half do any work
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex+i];
    __syncthreads();
    i /= 2;
  }
  // we store 1 number per block
  if (cacheIndex == 0)
    c[blockIdx.x] = cache[0];
}

int main() {
  float *a, *b;
  float c;
  float *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  // allocate memory on the CPU side
  a = (float*)malloc(N*sizeof(float));
  b = (float*)malloc(N*sizeof(float));
  // dimension: (number-of-blocks)
  partial_c = (float*)malloc(blocksPerGrid*sizeof(float)); // 1 float per block
  // allocate memory on GPU 
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));
  // fill in the host memory with data
  for (int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i*2;
  }
  // copy <a> and <b> to GPU
  HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

  dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

  // copy the array <c> back from GPU to CPU
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));
  // summing up partials to get result
  c = 0;
  for (int i=0; i<blocksPerGrid; i++) {
    c += partial_c[i];
  }
  // check calculation
  printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float(N-1))));

  // free GPU
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);
  // free CPU 
  free(a);
  free(b);
  free(partial_c);
  
  return 0;
}