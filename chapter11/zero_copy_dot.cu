#include "../common/book.h"

#define imin(a,b) (a<b?a:b)

const int N = 10*1024*1024; // 10MB 
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock); // max #blocks=32

// c: (num_blocks,)
__global__ void dot(int size, float* a, float* b, float* c) {
  // threads share 1 cache per block; cache: (256,)
  __shared__ float cache[threadsPerBlock]; 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIdx = threadIdx.x;
  float temp = 0;
  while(tid < size) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheIdx] = temp;
  // synchronize 256 threads in this block
  __syncthreads();
  // split half, add, split half, add... until only 1st number meaningful
  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIdx < i) {
      cache[cacheIdx] += cache[cacheIdx+i];
    }
    __syncthreads();
    i /= 2;
  }
  if (cacheIdx == 0) {
    c[blockIdx.x] = cache[0];
  }
}

// size: number of floats
float malloc_test(int size) { 
  cudaEvent_t start, stop;
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  float elapsedTime;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  // allocate memory on CPU
  a = (float*)malloc(size*sizeof(float));
  b = (float*)malloc(size*sizeof(float));
  partial_c = (float*)malloc(blocksPerGrid*sizeof(float)); // result: 1 block has 1 float

  // allocate memory on GPU
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));

  // fill in host memory with data
  for (int i=0; i<size; i++) {
    a[i] = i;
    b[i] = i*2;
  }
  // time record
  HANDLE_ERROR(cudaEventRecord(start, 0));
  // copy 'a' and 'b' to GPU
  HANDLE_ERROR(cudaMemcpy(dev_a, a, size*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size*sizeof(float), cudaMemcpyHostToDevice));
  // calculate
  dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
  // copy 'c' back from GPU to CPU
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));
  // stop timer
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  // summing up
  c = 0;
  for(int i=0; i<blocksPerGrid; i++) {
    c += partial_c[i];
  }
  // clean up
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_partial_c));
  
  free(a);
  free(b);
  free(partial_c);

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  printf("Value calculated: %f\n", c);

  return elapsedTime;
}

float cuda_host_alloc_test(int size) {
  cudaEvent_t start, stop;
  float *a, *b, c, *partial_c; // c: (1,); partial_c: (num_blocks,)
  float *dev_a, *dev_b, *dev_partial_c;
  float elapsedTime;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  // allocate on CPU - ZERO COPY
  HANDLE_ERROR(cudaHostAlloc(
    (void**)&a, 
    size*sizeof(float), 
    cudaHostAllocWriteCombined | cudaHostAllocMapped // NOTICE: cudaHostAllocWriteCombined is only for CPU->GPU direction
  ));
  HANDLE_ERROR(cudaHostAlloc(
    (void**)&b, 
    size*sizeof(float), 
    cudaHostAllocWriteCombined | cudaHostAllocMapped // NOTICE: cudaHostAllocWriteCombined is only for CPU->GPU direction
  ));
  HANDLE_ERROR(cudaHostAlloc(
    (void**)&partial_c, 
    blocksPerGrid*sizeof(float), 
    cudaHostAllocMapped // NOTICE: NO cudaHostAllocWriteCombined here because it's GPU->CPU direction
  ));
  // fill the host memory with data
  for (int i=0; i<size; i++) {
    a[i] = i;
    b[i] = i*2;
  }
  // helps GPU getting valid pointers of those areas
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));
  // start executing
  HANDLE_ERROR(cudaEventRecord(start, 0));
  dot<<<blocksPerGrid,threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
  HANDLE_ERROR(cudaThreadSynchronize());
  // stop timer
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  // finish up
  c = 0;
  for(int i=0; i<blocksPerGrid; i++) {
    c += partial_c[i];
  }
  // clean
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));
  HANDLE_ERROR(cudaFreeHost(partial_c));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  printf("Value calculated: %f\n", c);
  return elapsedTime;
}

int main() {
  cudaDeviceProp prop;
  int whichDevice;
  HANDLE_ERROR(cudaGetDevice(&whichDevice));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
  // capability for ZERO-COPY
  if (prop.canMapHostMemory != 1) {
    printf("Device cannot map memory.\n");
    return 0;
  }
  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  float elapsedTime = malloc_test(N);
  printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);

  elapsedTime = cuda_host_alloc_test(N);
  printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
}