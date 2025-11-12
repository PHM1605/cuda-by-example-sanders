#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_anim.h"

#define DIM 1024
#define PI 3.141592653
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

// copy SOME PIXELS in constant grid into input grid
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
// dstOut: which buffer to fetch data, texture-input or texture-output
__global__ void blend_kernel(float* dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut) {
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
  
  float t, l, c, r, b;
  if (dstOut) {
    t = tex1Dfetch<float>(texIn, top);
    l = tex1Dfetch<float>(texIn, left);
    c = tex1Dfetch<float>(texIn, offset);
    r = tex1Dfetch<float>(texIn, right);
    b = tex1Dfetch<float>(texIn, bottom);
  } else {
    t = tex1Dfetch<float>(texOut, top);
    l = tex1Dfetch<float>(texOut, left);
    c = tex1Dfetch<float>(texOut, offset);
    r = tex1Dfetch<float>(texOut, right);
    b = tex1Dfetch<float>(texOut, bottom);
  }
  
  dst[offset] = c + SPEED*(t + b + r + l - 4*c);
} 

struct DataBlock {
  unsigned char *output_bitmap; // GPU
  float *dev_inSrc;
  float *dev_outSrc;
  float *dev_constSrc;
  CPUAnimBitmap *bitmap; // CPU

  // modern texture objects
  cudaTextureObject_t texConstSrc;
  cudaTextureObject_t texIn;
  cudaTextureObject_t texOut;

  cudaEvent_t start, stop;
  float totalTime;
  float frames; // total number-of-frames up to now
};

// devPtr: pointer to list of float
static cudaTextureObject_t makeLinearFloatTex(void* devPtr, size_t sizeInBytes) {
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = devPtr;
  resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
  resDesc.res.linear.sizeInBytes = sizeInBytes;

  cudaTextureDesc texDesc = {};
  texDesc.readMode = cudaReadModeElementType;

  cudaTextureObject_t tex = 0;
  HANDLE_ERROR(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
  return tex;
}

void anim_gpu(DataBlock *d, int ticks) {
  HANDLE_ERROR(cudaEventRecord(d->start, 0));
  dim3 blocks(DIM/16, DIM/16);
  dim3 threads(16, 16);
  CPUAnimBitmap* bitmap = d->bitmap;

  volatile bool dstOut = true;
  for (int i=0; i<90; i++) {
    float *in, *out;
    if (dstOut) {
      in = d->dev_inSrc;
      out = d->dev_outSrc;
    } else {
      out = d->dev_inSrc;
      in = d->dev_outSrc;
    }
    copy_const_kernel<<<blocks,threads>>>(in, d->texConstSrc); // keep SOME PIXELS as torch sources fixed; not overwritting entire grid but only some pixels
    // if dstOut == True => <texIn> is important; otherwise <texOut>
    blend_kernel<<<blocks,threads>>>(out, dstOut, d->texIn, d->texOut);
    dstOut = !dstOut;
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
  cudaDestroyTextureObject(d->texConstSrc);

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

  const size_t RGBA_BYTES = bitmap.image_size(); // DIM*DIM*4
  const size_t GRID_BYTES = DIM*DIM*sizeof(float); // DIM*DIM*size(float)

  // allocate memory for output bitmap on GPU
  HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, RGBA_BYTES));
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, GRID_BYTES));
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, GRID_BYTES));
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, GRID_BYTES));

  // fill a random CPU bitmap
  float *temp = (float*)malloc(GRID_BYTES);
  for (int i=0; i<DIM*DIM; ++i) 
    temp[i] = 0.0f;

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
  HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, GRID_BYTES, cudaMemcpyHostToDevice));

  // Set another map of heat
  // row [800:1023] col[0:199]
  for (int y=800; y<DIM; y++) {
    for (int x=0; x<200; x++) {
      temp[x+y*DIM] = MAX_TEMP;
    }
  }
  HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, GRID_BYTES, cudaMemcpyHostToDevice));
  // clear outSrc
  HANDLE_ERROR(cudaMemset(data.dev_outSrc, 0, GRID_BYTES));

  free(temp);

  // create texture objects
  data.texConstSrc = makeLinearFloatTex(data.dev_constSrc, GRID_BYTES);
  data.texIn = makeLinearFloatTex(data.dev_inSrc, GRID_BYTES);
  data.texOut = makeLinearFloatTex(data.dev_outSrc, GRID_BYTES);

  bitmap.anim_and_exit(
    (void (*)(void*, int))anim_gpu,
    (void (*)(void*))anim_exit
  );
  return 0;
}
