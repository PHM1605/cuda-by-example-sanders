#include "../common/book.h"

#define N 10

int main() {
  int a[N], b[N], c[N];
  // fill the arrays 'a' and 'b' on the CPU
  for (int i=0; i<N; i++) {
    a[i] = -i;
    b[i] = i*i;
  }
  
  int* dev_a, *dev_b, *dev_c;


  return 0;
}