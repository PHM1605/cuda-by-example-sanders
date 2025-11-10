#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024
#define PI 3.14159265

__global__ void kernel(unsigned char *ptr) {
  // map from threadIdx & blockIdx to pixel position x,y on WHOLE IMAGE
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*blockDim.x*gridDim.x;
  // shared memory among threads (1 block has 1 shared var)
  __shared__ float shared[16][16];
  // each thread computes a value
  const float period = 128.0f;
  shared[threadIdx.y][threadIdx.x] = 255 * (sinf(x*2.0f*PI/ period) + 1.0f) *(sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;
  
  // NOTE: if we remove this, some threads might exit earlier and draw wrong
  __syncthreads();

  // fill in the big image
  ptr[offset*4+0] = 0;
  ptr[offset*4+1] = shared[15-threadIdx.x][15-threadIdx.y];
  ptr[offset*4+2] = 0;
  ptr[offset*4+3] = 255;
}

struct DataBlock {
  unsigned char *dev_bitmap;
};

int main() {
  DataBlock data;
  CPUBitmap bitmap(DIM, DIM, &data);
  unsigned char *dev_bitmap;
  HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
  data.dev_bitmap = dev_bitmap;

  // each big cell is 16x16 (threads)
  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16, 16);

  kernel<<<grids, threads>>>(dev_bitmap);

  // notice: some threads may reach here BEFORE other threads when __syncthreads() missing
  HANDLE_ERROR(cudaMemcpy(
    bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost
  ));
  HANDLE_ERROR(cudaFree(dev_bitmap));
  bitmap.display_and_exit();

  return 0;
}