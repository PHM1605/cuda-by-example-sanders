#include "../common/book.h"
#include "../common/gpu_anim.h"

#define DIM 1024
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f 

struct DataBlock {
  float *dev_inSrc;
  float *dev_outSrc;
  float *dev_constSrc;

  // modern textures objects
  cudaTextureObject_t texConstSrc;
  cudaTextureObject_t texIn;
  cudaTextureObject_t texOut;

  cudaEvent_t start, stop;
  float totalTime;
  float frames;
};

// copy SOME PIXELS in constant grid into input grid
__global__ void copy_const_kernel(float* iptr, cudaTextureObject_t texConst) {
  // map from threadIdx & blockIdx to pixel position
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*gridDim.x*blockDim.x;
  // fetch texture at that position; copy non-zero pixels only
  float c = tex1Dfetch<float>(texConst, offset);
  if (c!=0) {
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
  if (x==0) left++;
  if (x==DIM-1) right--;

  int top = offset - DIM;
  int bottom = offset + DIM;
  if (y==0) top += DIM;
  if (y==DIM-1) bottom -= DIM;

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
  dst[offset] = c + SPEED*(t+b+r+l-4*c);
}

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

// OpenGL uses vector of 4 chars (uchar4*) instead of usual char*
void anim_gpu(uchar4* outputBitmap, DataBlock* d, int ticks) {
  HANDLE_ERROR(cudaEventRecord(d->start, 0));
  dim3 blocks(DIM/16, DIM/16);
  dim3 threads(16, 16);

  // dstOut determin which texture (dev_inSrc/texIn) or (dev_outSrc/texOut)
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
    // overwrite SOME pixels in constant grid only, not entire grid
    copy_const_kernel<<<blocks,threads>>>(in, d->texConstSrc);
    blend_kernel<<<blocks,threads>>>(out, dstOut, d->texIn, d->texOut);
    dstOut = !dstOut;
  }

  float_to_color<<<blocks,threads>>>(outputBitmap, d->dev_inSrc);

  // record time
  HANDLE_ERROR(cudaEventRecord(d->stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(d->stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
  d->totalTime += elapsedTime;
  ++d->frames;
  printf("Average Time per frame: %3.1f ms\n", d->totalTime/d->frames);
}

void anim_exit(DataBlock *d) {
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
  GPUAnimBitmap bitmap(DIM, DIM, &data);
  data.totalTime = 0;
  data.frames = 0;

  // prepare time 
  HANDLE_ERROR(cudaEventCreate(&data.start));
  HANDLE_ERROR(cudaEventCreate(&data.stop));
  
  const size_t RGBA_BYTES = bitmap.image_size(); // DIM*DIM*4
  const size_t GRID_BYTES = DIM*DIM*sizeof(float); // DIM*DIM*size(float)

  // allocate memory for bitmaps on GPU
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, RGBA_BYTES));
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, GRID_BYTES));
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, GRID_BYTES));

  // create texture objects
  data.texConstSrc = makeLinearFloatTex(data.dev_constSrc, GRID_BYTES);
  data.texIn = makeLinearFloatTex(data.dev_inSrc, GRID_BYTES);
  data.texOut = makeLinearFloatTex(data.dev_outSrc, GRID_BYTES);

  // Initialize some random data 
  float *temp = (float*)malloc(GRID_BYTES);
  for (int i=0; i<DIM*DIM; ++i) 
    temp[i] = 0.0f;
  for (int y=310; y<601; y++) {
    for (int x=300; x<610; x++) {
      temp[y*DIM+x] = MAX_TEMP;
    }
  }
  // single hot points
  temp[DIM*100+100] = (MAX_TEMP+MIN_TEMP)/2;
  temp[DIM*700+100] = MIN_TEMP;
  temp[DIM*300+300] = MIN_TEMP;
  temp[DIM*200+700] = MIN_TEMP;
  for (int y=800; y<900; y++) {
    for (int x=400; x<500; x++) {
      temp[x+y*DIM] = MIN_TEMP;
    }
  }
  HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, GRID_BYTES, cudaMemcpyHostToDevice));

  // set another map of heat
  for (int y=800; y<DIM; y++) {
    for (int x=0; x<200; x++) {
      temp[x+y*DIM] = MAX_TEMP;
    }
  }
  HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, GRID_BYTES, cudaMemcpyHostToDevice));
  // clear outSrc
  HANDLE_ERROR(cudaMemset(data.dev_outSrc, 0, GRID_BYTES));

  free(temp);

  bitmap.anim_and_exit(
    (void (*)(uchar4*, void*, int))anim_gpu,
    (void (*)(void*))anim_exit
  );

  return 0;
}