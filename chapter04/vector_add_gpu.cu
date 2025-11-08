#include "../common/book.h"

#define N 50000

__global__ void add(int* a, int *b, int *c) {
  int tid = blockIdx.x; // thread index
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

int main() {
  int a[N], b[N], c[N];
  // fill the arrays 'a' and 'b' on the CPU
  for (int i=0; i<N; i++) {
    a[i] = -i;
    b[i] = i*i;
  }
  
  int* dev_a, *dev_b, *dev_c;
  // allocate memory on GPU
  HANDLE_ERROR( cudaMalloc((void**)&dev_a, N*sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&dev_b, N*sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&dev_c, N*sizeof(int)) );
  // copy the arrays 'a' and 'b' to the GPU
  HANDLE_ERROR( cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice) );
  HANDLE_ERROR(cudaMemcpy(dev_b ,b, N*sizeof(int), cudaMemcpyHostToDevice));

  add<<<N, 1>>>(dev_a, dev_b, dev_c);
  
  // copy result back to cpu
  HANDLE_ERROR( cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost ));
  // display the result 
  for (int i=0; i<N; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }
  // free
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}