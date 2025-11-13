#include "../common/book.h"

#define SIZE (10*1024*1024) // 10MB of data

// CPU uses usual <malloc> 
float cuda_malloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsedTime;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  a = (int*)malloc(size*sizeof(*a));
  HANDLE_NULL(a);
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size*sizeof(*dev_a)));

  // up: decide which location we copy data
  // record data copying in two directions
  HANDLE_ERROR(cudaEventRecord(start , 0));
  for (int i=0; i<100; i++) {
    // host to device
    if (up) {
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size*sizeof(*dev_a), cudaMemcpyHostToDevice));
    } 
    // device to host
    else {
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size*sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  }
  // clean and return time
  free(a);
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return elapsedTime;
}

// CPU uses pinned buffer <cudaHostAlloc>
float cuda_host_alloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsedTime;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  HANDLE_ERROR(cudaHostAlloc((void**)&a, size*sizeof(int), cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size*sizeof(int)));

  HANDLE_ERROR(cudaEventRecord(start, 0));
  for (int i=0; i<100; i++) {
    if (up) { 
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size*sizeof(int), cudaMemcpyHostToDevice));
    } else {
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size*sizeof(int), cudaMemcpyDeviceToHost));
    }
  }
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  // clean
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return elapsedTime;
}

int main() {
  float elapsedTime;
  // we create 100 times of SIZE*sizeof(int) [bytes]
  float MB = (float)100*SIZE*sizeof(int)/1024/1024;
  
  // up direction (cpu->gpu) of usual <malloc>
  elapsedTime = cuda_malloc_test(SIZE, true); // [ms]
  printf("Time using malloc: %3.1f ms\n", elapsedTime);
  printf("\tMB/s during copy up: %3.1f\n", MB/(elapsedTime/1000));
  // down direction (gpu->cpu) of usual <malloc>
  elapsedTime = cuda_malloc_test(SIZE, false); // [ms]
  printf("Time using malloc: %3.1f ms\n", elapsedTime);
  printf("\tMB/s during copy down: %3.1f\n", MB/(elapsedTime/1000));
  
  // up direction (cpu-pinned->gpu) of <host-alloc>
  elapsedTime = cuda_host_alloc_test(SIZE, true);
  printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
  printf("\tMB/s during copy up: %3.1f\n", MB/(elapsedTime/1000));
  // down direction (gpu->cpu-pinned) of <host-alloc>
  elapsedTime = cuda_host_alloc_test(SIZE, false);
  printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
  printf("\tMB/s during copy down: %3.1f\n", MB/(elapsedTime/1000));

  return 0;
}