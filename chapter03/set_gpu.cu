#include "../common/book.h"

int main(void) {
    cudaDeviceProp prop;
    int dev;
    HANDLE_ERROR( cudaGetDevice(&dev) );
    printf("ID of current Cuda device: %d\n", dev);

    // set config for which cuda device we want to pick out
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;

    // set to the collected device
    HANDLE_ERROR( cudaChooseDevice(&dev, &prop) );
    printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
    

    return 0;
}