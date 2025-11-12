#define GL_GLEXT_PROTOTYPES
#include "GL/glut.h"
#include "cuda.h"
#include "cuda_gl_interop.h"

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 512

// data to be shared between OpenGL and Cuda
GLuint bufferObj; // OpenGL buffer object handle
cudaGraphicsResource *resource; // Cuda

__global__ void kernel(uchar4* ptr) {
  // map from threadIdx/blockIdx to pixel position
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*blockDim.x*gridDim.x;

  // calculate some values at that position; to change range to [-0.5,0.5]
  float fx = x/(float)DIM - 0.5f;
  float fy = y/(float)DIM - 0.5f;
  unsigned char green = 128 + 127*sin(abs(fx*100)-abs(fy*100));

  // set image value
  ptr[offset].x = 0;
  ptr[offset].y = green;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}

static void draw_func() {
  // usually last parameter is for buffer pointer; we bind to Cuda <resource> already, so this field becomes "offset to buffer"
  glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0); 
  glutSwapBuffers();
}

static void key_func(unsigned char key, int x, int y) {
  switch(key) {
    // when user presses <Esc> then clean up and exit program
    case 27:
      // clean up Cuda
      HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
      // clean up OpenGL
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      glDeleteBuffers(1, &bufferObj);
      // exit
      exit(0);
  }
}

int main(int argc, char** argv) {
  // init
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(DIM, DIM);
  glutCreateWindow("bitmap");
  // choose a device with this property (in this case, at least 1.0 version)
  cudaDeviceProp prop;
  int dev;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;
  HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
  // set this device to be used for OpenGL
  HANDLE_ERROR(cudaSetDevice(dev));
  // create buffer object and store in handle <bufferObj>
  glGenBuffers(1, &bufferObj); // create handle
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj); // bind handle to buffer
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM*4, NULL, GL_DYNAMIC_DRAW_ARB); // create buffer of size DIM*DIM*4 and init value NULL
  // inform Cuda to use this buffer on OpenGL via Cuda-friendly <resource> handle
  HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

  // create a pointer <devPtr> to this area (from <resource>) FOR CUDA to use later
  uchar4 *devPtr;
  size_t size;
  HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL)); // create 1 pointer, handled by <resource>, and NOT streaming
  HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

  // first, ask Cuda to generate image
  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16, 16);
  kernel<<<grids,threads>>>(devPtr);
  // disconnect Cuda from buffer
  HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));
  // from now on OpenGL will handle
  glutKeyboardFunc(key_func); // simple keyboard function for now: when user press <Esc> then exit program
  glutDisplayFunc(draw_func); 
  glutMainLoop();

  return 0;
}