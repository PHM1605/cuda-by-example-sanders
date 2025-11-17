#ifndef __LOCK_H__
#define __LOCK_H__

#include "../common/book.h"
#include <cuda_runtime.h>

struct Lock {
  int *mutex;

  Lock(void) {
    int state = 0;
    HANDLE_ERROR(cudaMalloc((void**)&mutex, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
  }

  ~Lock(void) {
    cudaFree(mutex);
  }

  __device__ void lock(void) {
    // CAS: Compare And Swap
    // check: position at position <mutex> equals <0> or not. If yes then set to <1>
    // atomicCAS() returns the original value at <*mutex> 
    // <*mutex>=0 means FREE; <mutex>=1 means LOCKED
  #ifdef __CUDA_ARCH__
    while(atomicCAS(mutex, 0, 1) != 0);
  #endif
  }

  __device__ void unlock() {
#ifdef __CUDA_ARCH__
    atomicExch(mutex, 0);
#endif
  }
};

#endif 