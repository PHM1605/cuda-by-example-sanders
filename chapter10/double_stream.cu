#include "../common/book.h"

#define N (1024*1024) // 1MB data
#define FULL_DATA_SIZE (N*20) // 20MB data

__global__ void kernel(int *a, int *b, int *c) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx<N) {
    // 1 block = 256 threads; averaging with wrap-around within block
    int idx1 = (idx + 1) % 256; 
    int idx2 = (idx + 2) % 256;
    float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
    float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
    c[idx] = (as+bs) / 2;
  }
}

int main() {
  cudaDeviceProp prop;
  int whichDevice;
  HANDLE_ERROR(cudaGetDevice(&whichDevice));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
  // device overlap: copy GPU-CPU AND kernel execution at the same time
  if (!prop.deviceOverlap) {
    printf("Device will not handle overlaps, so no speed up from streams\n");
    return 0;
  }
  
  // calculate time
  cudaEvent_t start, stop;
  float elapsedTime;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  cudaStream_t stream0, stream1;
  HANDLE_ERROR(cudaStreamCreate(&stream0));
  HANDLE_ERROR(cudaStreamCreate(&stream1));

  // data allocation
  int *host_a, *host_b, *host_c;
  int *dev_a0, *dev_b0, *dev_c0; // GPU buffers for stream0
  int *dev_a1, *dev_b1, *dev_c1; // GPU buffers for stream1

  // allocate GPU of 1MB each for stream0
  HANDLE_ERROR(cudaMalloc((void**)&dev_a0, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b0, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c0, N*sizeof(int)));

  // allocate GPU of 1MB each for stream1
  HANDLE_ERROR(cudaMalloc((void**)&dev_a1, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b1, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c1, N*sizeof(int)));

  // allocate pinned-buffer CPU of 20MB each; required for STREAM
  HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault));
  for (int i=0; i<FULL_DATA_SIZE; i++) {
    host_a[i] = rand();
    host_b[i] = rand();
  }



  
  // go over full data 20MB in SMALLER CHUNKS of 1MB each
  // NOTE: 1st hunk's async 1, async 2, async 3 and async 4; then next chunk's asyncs are put on <stream>
  for (int i=0; i<FULL_DATA_SIZE; i+=N) {
    HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream)); // async 1
    HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream)); // async 2

    kernel<<<N/256,256,0,stream>>>(dev_a, dev_b, dev_c); // async 3

    HANDLE_ERROR(cudaMemcpyAsync(host_c+i, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost, stream)); // async 4
  }
  // copy result from <pinned-buffer> to normal CPU buffer
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  // time recording 
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time taken: %3.1f ms\n", elapsedTime);

  // clean up
  HANDLE_ERROR(cudaFreeHost(host_a));
  HANDLE_ERROR(cudaFreeHost(host_b));
  HANDLE_ERROR(cudaFreeHost(host_c));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  HANDLE_ERROR(cudaStreamDestroy(stream));

  return 0;
}