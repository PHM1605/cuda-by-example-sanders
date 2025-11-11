#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_anim.h"

#define DIM 1024
#define PI 3.141592653
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

// copy constant grid into input grid
__global__ void copy_const_kernel(float* iptr, cudaTextureObject_t texConst) {
  // map from threadIdx & blockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  // fetch texture at that position
  float c = tex1Dfetch<float>(texConst, offset);
  // don't fetch texture with 0 value
  if (c != 0) {
    iptr[offset] = c;
  }
}

// each thread (= each pixel) performs temperature calculation
__global__ void blend_kernel(float *outSrc, const float *inSrc) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*blockDim.x*gridDim.x;

  int left = offset - 1;
  int right = offset + 1;
  if (x==0) left++; // for point in left edge => left=0
  if (x==DIM-1) right--; // for point in right edge => right=DIM-1

  int top = offset - DIM; // point above that location
  int bottom = offset + DIM; // point below that location
  if (y==0) top += DIM; // point in first row => top=that point
  if (y==DIM-1) bottom -= DIM; // point in last row => below=that point
  outSrc[offset] = inSrc[offset] + SPEED*(inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - 4*inSrc[offset]);
} 

struct DataBlock {
  unsigned char *output_bitmap; // GPU
  float *dev_inSrc;
  float *dev_outSrc;
  float *dev_constSrc;
  CPUAnimBitmap *bitmap; // CPU

  // modern texture objects
  cudaTextureObject_t textConstSrc;
  cudaTextureObject_t texIn;
  cudaTextureObject_t texOut;

  cudaEvent_t start, stop;
  float totalTime;
  float frames; // total number-of-frames up to now
};

void anim_gpu(DataBlock *d, int ticks) {
  HANDLE_ERROR(cudaEventRecord(d->start, 0));
  dim3 blocks(DIM/16, DIM/16);
  dim3 threads(16, 16);
  CPUAnimBitmap* bitmap = d->bitmap;

  for (int i=0; i<90; i++) {
    copy_const_kernel<<<blocks,threads>>>(d->dev_inSrc, d->dev_constSrc);
    blend_kernel<<<blocks,threads>>>(d->dev_outSrc, d->dev_inSrc);
    swap(d->dev_inSrc, d->dev_outSrc); // <d->dev_inSrc> now used for next step; <d->dev_outSrc> is outdated  
  }
  float_to_color<<<blocks,threads>>>(d->output_bitmap, d->dev_inSrc); // from array of floats => color image
  // move resulting image from GPU to CPU
  HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));

  // time-related handling
  HANDLE_ERROR(cudaEventRecord(d->stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(d->stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
  d->totalTime += elapsedTime;
  ++d->frames;
  printf("Average time per frame: %3.1f ms\n", d->totalTime/d->frames);
}

void anim_exit(DataBlock* d) {
  cudaDestroyTextureObject(d->texIn);
  cudaDestroyTextureObject(d->texOut);
  cudaDestroyTextureObject(d->textConstSrc);

  HANDLE_ERROR(cudaFree(d->dev_inSrc));
  HANDLE_ERROR(cudaFree(d->dev_outSrc));
  HANDLE_ERROR(cudaFree(d->dev_constSrc));

  HANDLE_ERROR(cudaEventDestroy(d->start));
  HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main() {
  DataBlock data;
  CPUAnimBitmap bitmap(DIM, DIM, &data);
  data.bitmap = &bitmap;
  data.totalTime = 0;
  data.frames = 0;
  HANDLE_ERROR(cudaEventCreate(&data.start));
  HANDLE_ERROR(cudaEventCreate(&data.stop));
  int imageSize = bitmap.image_size();

  // allocate memory for output bitmap on GPU
  HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, imageSize));

  HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, DIM*DIM*sizeof(float))); // 1 float = 4 bytes
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, DIM*DIM*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, DIM*DIM*sizeof(float)));

  // fill a random CPU bitmap
  float *temp = (float*)malloc(DIM*DIM*sizeof(float));
  for (int i=0; i<DIM*DIM; ++i) 
    temp[i] = 0.0f;

  // rectangle of heat: row [311,600]; col [301,599]
  
  for (int i=0; i<DIM*DIM; i++) {
    temp[i] = 0; // temp[0]: first 4 bytes; temp[1]: second 4 bytes; ...
    // int x = i%DIM;
    // int y = i/DIM;

    // Set the map of heat
    // x: [300,610]; y: [310,600]
    for (int y=310; y<601; y++) {
      for (int x=300; x<=610; x++) {
        temp[y*DIM+x] = MAX_TEMP;
      }
    }
    
    // Single hot points
    temp[DIM*100+100] = (MAX_TEMP+MIN_TEMP)/2.0f;
    temp[DIM*700+100] = MIN_TEMP;
    temp[DIM*300+300] = MIN_TEMP;
    temp[DIM*200+700] = MIN_TEMP;
    // row [801:899]; col [401:499]
    for (int y=800; y<900; y++){
      for (int x=400; x<500; x++) {
        temp[x+y*DIM] = MIN_TEMP;
      }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice));

    // Set another map of heat
    // row [800:1023] col[0:199]
    for (int y=0; y<DIM; y++) {
      for (int x=0; x<200; x++) {
        temp[x+y*DIM] = MAX_TEMP;
      }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice));

    free(temp);
    bitmap.anim_and_exit(
      (void (*)(void*, int))anim_gpu,
      (void (*)(void*))anim_exit
    );
  }

  return 0;
}
