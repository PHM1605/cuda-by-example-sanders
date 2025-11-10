#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define INF 2e10f
#define SPHERES 20
#define rnd(x) (x*rand()/RAND_MAX) // random between 0 and x
#define DIM 1024 

struct Sphere {
  float r, g, b;
  float radius;
  float x, y, z; // sphere center location
  // if the pixel at (ox,oy) hits the sphere => return distance; else -INF
  // if hits more than 1 sphere => return only the furthest from camera (i.e. nearest from us)
  __device__ float hit(float ox, float oy, float* n) {
    // distance from CENTER of sphere (x,y) to a pixel (ox,oy)
    float dx = ox - x;
    float dy = oy - y;
    // draw xy-plane to see
    // if z-axis crossing that pixel penetrates circle
    if (dx*dx+dy*dy < radius*radius) 
    {
      // draw x-z plane to understand
      float dz = sqrtf(radius*radius-dx*dx-dy*dy);
      *n = dz / radius;
      return z+dz; // distance from that point to the furthest point on sphere
    }
    return -INF;
  }
};

__global__ void kernel(Sphere *s, unsigned char* ptr) {
  // map from threadIdx/blockIdx to pixel position (x,y)
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*blockDim.x*gridDim.x;
  // shifted pixel position to center 
  float ox = (x-DIM/2);
  float oy = (y-DIM/2);

  float maxz = -INF;
  float r=0, g=0, b=0;
  for (int i=0; i<SPHERES; i++) {
    // <brightness> to adjust brightness; light-ray through center will be brightest
    float fscale;
    float dist = s[i].hit(ox, oy, &fscale);
    if (dist>maxz) {
      r = s[i].r * fscale;
      g = s[i].g * fscale;
      b = s[i].b * fscale;
      maxz = dist;
    }
  }

  ptr[offset*4+0] = (int)(r*255);
  ptr[offset*4+1] = (int)(g*255);
  ptr[offset*4+2] = (int)(b*255);
  ptr[offset*4+3] = 255;
}

int main() {
  // capture the start/stop time 
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start)); // create start-time object
  HANDLE_ERROR(cudaEventCreate(&stop)); // create stop-time object
  HANDLE_ERROR(cudaEventRecord(start, 0)); // note the start timestamp in stream 0

  CPUBitmap bitmap(DIM, DIM);
  // data.bitmap = &bitmap;
  unsigned char *dev_bitmap;
  Sphere *s; // for the list of Spheres on GPU

  // allocate memory on GPU for the output bitmap
  HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
  // allocate memory on GPU for the Sphere dataset
  HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere)*SPHERES));

  // allocate sphere memory on CPU, generate some random values
  Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
  for (int i=0; i<SPHERES; i++) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    // random center at (-500,500) range each of xyz
    temp_s[i].x = rnd(1000.0f) - 500;
    temp_s[i].y = rnd(1000.0f) - 500;
    temp_s[i].z = rnd(1000.0f) - 500;
    // random radius from 20 to 120
    temp_s[i].radius = rnd(100.0f) + 20; 
  }

  // copy Spheres data to GPU
  HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice));
  free(temp_s);

  // resulting image on GPU
  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16, 16);
  kernel<<<grids, threads>>>(s, dev_bitmap);

  // copy resulting image from GPU to CPU
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
  bitmap.display_and_exit();

  // cleaning
  cudaFree(dev_bitmap);
  cudaFree(s);
}