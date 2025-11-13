#include "../common/book.h"

#define SIZE (100*1024*1024)

int main() {
  // time record preparation
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));
  // generate 100MB random data
  unsigned char *buffer = (unsigned char*)big_random_block(SIZE); 
  // char is 8bit => 0x00 to 0xFF => we need 256 bins
  unsigned int histo[256];
  for (int i=0; i<256; i++) {
    histo[i] = 0;
  }
  // assign occurence
  for (int i=0; i<SIZE; i++) {
    histo[buffer[i]]++;
  }
  // record time
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time to generate: %3.1f ms\n", elapsedTime);
  // debugging
  long histoCount = 0;
  for (int i=0; i<256; i++) {
    histoCount += histo[i];
  }
  printf("Histogram sum: %ld\n", histoCount);

  // cleaning 
  
  free(buffer);
  return 0;
}