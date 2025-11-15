#include "../common/book.h"

#define imin(a,b) (a<b?a:b)

#define N (33*1024*1024) // 33MB data
const int threadsPerBlock = 256;
// maximum 32 blocks per GPU
// N will be split between 2 GPUs
const int blocksPerGrid = imin(32, (N/2+threadsPerBlock-1)/threadsPerBlock); 

__global__ void dot(int size, float *a, float *b, float *c) {

}

struct DataStruct {
  int deviceID;
  int size;
  float *a;
  float *b;
  float returnValue;
};

int main() {
  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);
    return 0;
  }
  // allocate and fill CPU with data
  float *a = (float*)malloc(sizeof(float) * N);
  HANDLE_NULL(a);
  float *b = (float*)malloc(sizeof(float)*N);
  HANDLE_NULL(b);
  
  for (int i=0; i<N; i++) {
    a[i] = i;
    b[i] = i*2;
  }
}