#include "../common/book.h"
#include "../common/gpu_anim.h"

#define DIM 1024

__global__ void kernel(uchar4 *ptr, int ticks) {
  // map from threadIdx/ blockIdx to pixel position
  
}

int main() {
  GPUAnimBitmap bitmap(DIM, DIM, NULL);

  return 0;
}