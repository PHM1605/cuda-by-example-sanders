#include "../common/book.h"

int main() {
  // Check current Cuda device
  int dev;
  HANDLE_ERROR( cudaGetDevice(&dev) );
  printf("ID of current CUDA device: %d\n", dev);

  // Change current Cuda device
  cudaDeviceProp prop;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 3;
  HANDLE_ERROR( cudaChooseDevice(&dev, &prop) );
  printf("ID of Cuda device closest to revision 1.3: %d\n", dev);
  HANDLE_ERROR( cudaSetDevice(dev) );

  return 0;
}