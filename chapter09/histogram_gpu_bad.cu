// This approach is bad because every threads access small GLOBAL <histogram> buffer
#include "../common/book.h"

#define SIZE (100*1024*1024)

__global__ void histo_kernel(
  unsigned char *buffer, // 100MB data 
  long size, // size of <buffer>
  unsigned int *histo // output histogram
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // blockDim.x = 256
  int stride = blockDim.x * gridDim.x; // blockDim.x = 56
  while (i < size) {
    atomicAdd(&(histo[buffer[i]]), 1);
    i += stride;
  }
}

int main(){
  // time record preparation
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));
  
  // generate 100MB random data
  unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
  
  // allocate memory on GPU for the file's data of 100MB
  unsigned char *dev_buffer;
  HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));
  HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));
  
  // allocate memory on GPU for histogram; init to 0 all
  unsigned int *dev_histo;
  HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256*sizeof(int)));
  HANDLE_ERROR(cudaMemset(dev_histo, 0, 256*sizeof(int)));
  
  // rule of thumb: use <number of processors*2> blocks
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount*2;
  histo_kernel<<<blocks,256>>>(dev_buffer, SIZE, dev_histo);

  // copy computed-histogram from GPU to CPU
  unsigned int histo[256];
  HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256*sizeof(int), cudaMemcpyDeviceToHost));

  // record time
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time to generate: %3.1f ms\n", elapsedTime);
  

  // debugging the GPU->CPU histogram with HISTOGRAM COUNT 
  long histoCount = 0;
  for (int i=0; i<256; i++) {
    histoCount += histo[i];
  }
  printf("Histogram sum: %ld\n", histoCount);

  // debugging by DECREASE GPU COUNT when CPU meets a char => good if all bins = 0 at the end
  for (int i=0; i<SIZE; i++)
    histo[buffer[i]]--;
  for (int i=0; i<256; i++) {
    if (histo[i] != 0)
      printf("Failure at %d!\n", i);
  }

  // cleaning
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  cudaFree(dev_histo);
  cudaFree(dev_buffer);
  free(buffer);

  return 0;
}