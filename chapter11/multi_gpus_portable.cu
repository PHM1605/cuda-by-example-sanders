#include "../common/book.h"
#include <chrono>

#define imin(a,b) (a<b?a:b)

#define N (10*1024*1024) // 33MB data
const int threadsPerBlock = 256;
// maximum 32 blocks per GPU
// N will be split between 2 GPUs
const int blocksPerGrid = imin(32, (N/2+threadsPerBlock-1)/threadsPerBlock); 

__global__ void dot(int size, float *a, float *b, float *c) {
  // cache: (num_threads,)
  __shared__ float cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int cacheIdx = threadIdx.x;

  float temp = 0;
  while (tid < size) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheIdx] = temp;

  // sync threads in this block
  __syncthreads();
  // block reduction: add-up all threads
  int i = blockDim.x/2;
  while (i!=0) {
    if (cacheIdx < i) {
      cache[cacheIdx] += cache[cacheIdx + i];
    }
    __syncthreads();
    i /= 2;
  }
  if (cacheIdx == 0) {
    c[blockIdx.x] = cache[0];
  }
}

struct DataStruct {
  int deviceID;
  int size;
  int offset;
  float *a;
  float *b;
  float returnValue;
};

void* routine(void *pvoidData) {
  DataStruct *data = (DataStruct*)pvoidData;
  // We have already set Device 0 when set PORTABLE host memory
  if (data->deviceID != 0) {
    // host PORTABLE PINNED memory for device 1
    HANDLE_ERROR(cudaSetDevice(data->deviceID)); 
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost)); 
  }
  
  int size = data->size; // N/2 if we use 2 GPUs
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  // CPU side
  a = data->a;
  b = data->b;
  partial_c = (float*)malloc(blocksPerGrid*sizeof(float)); // (number-of-blocks,) floats
  // CPU side but PORTABLE-PINNED
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
  // GPU side 
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));
  
  dev_a += data->offset;
  dev_b += data->offset;

  // kernel execution
  dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
  // copy <c> back to CPU
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));
  // add up
  c = 0;
  for (int i=0; i<blocksPerGrid; i++) {
    c += partial_c[i];
  }
  // clean up
  HANDLE_ERROR(cudaFree(dev_partial_c));
  free(partial_c);
  data->returnValue = c;
  return 0;
}

int main() {
  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);
    return 0;
  }

  // check if pinned-memory support
  cudaDeviceProp prop;
  for (int i=0; i<2; i++) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    if (prop.canMapHostMemory != 1) {
      printf("Device %d cannot map memory.\n", i);
      return 0;
    }
  }

  // before setting PORTABLE host memory, we must set 1 of the 2 CUDA device
  float *a, *b;
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  HANDLE_ERROR(cudaHostAlloc(
    (void**)&a, 
    N*sizeof(float),
    cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped
  ));
  HANDLE_ERROR(cudaHostAlloc(
    (void**)&b,
    N*sizeof(float),
    cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped
  ));
  // fill PORTABLE memory with data
  for (int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i*2;
  }

  // Setup 2 GPUs
  DataStruct data[2];
  data[0].deviceID = 0;
  data[0].offset = 0;
  data[0].size = N/2;
  data[0].a = a;
  data[0].b = b;

  data[1].deviceID = 1;
  data[1].offset = N/2;
  data[1].size = N/2;
  data[1].a = a;
  data[1].b = b;

  // timer
  auto t0 = std::chrono::high_resolution_clock::now();

  // Each GPU is managed by 1 CPU thread
  CUTThread thread = start_thread(routine, &(data[1])); // default thread
  routine(&(data[0])); // one additional thread
  end_thread(thread); // main thread waits for the other thread to finish

  // timer stop
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsedTime = std::chrono::duration<double, std::milli>(t1-t0).count();

  // clean up PORTABLE-PINNED and display result
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));
  printf("Value calculated: %f\n", data[0].returnValue+data[1].returnValue);
  printf("Time elapsed: %3.1f ms\n", elapsedTime);

  return 0;
}