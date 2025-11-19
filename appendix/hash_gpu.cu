#include "../common/book.h"
#include "lock.h"

#define HASH_ENTRIES 1024 // number of buckets
#define SIZE (10*1024*1024) // total size of all random numbers generated
#define ELEMENTS (SIZE/sizeof(unsigned int)) // number of random numbers generated

struct Entry {
  unsigned int key;
  void* value;
  Entry *next;
};

struct Table {
  size_t count; // number of buckets
  Entry **entries; // array of pool indices (into pool); -1 = empty
  Entry *pool; // all Entries
};

// we can perform hash function either on CPU or GPU
__device__ __host__ size_t hash(unsigned int value, size_t count) {
  return value % count;
}

void initialize_table(Table& table, int entries, int elements) {
  table.count = entries;
  HANDLE_ERROR(cudaMalloc((void**)&table.entries, entries*sizeof(Entry*)));
  HANDLE_ERROR(cudaMemset(table.entries, 0, entries*sizeof(Entry*))); // set unset buckets to -1
  HANDLE_ERROR(cudaMalloc((void**)&table.pool, elements*sizeof(Entry)));
}

void free_table(Table& table) {
  HANDLE_ERROR(cudaFree(table.pool));
  HANDLE_ERROR(cudaFree(table.entries));
}

// copy table from GPU to CPU
void copy_table_to_host(const Table &dev_table, Table &host_table) {
  host_table.count = dev_table.count;
  host_table.entries = (Entry**)calloc(dev_table.count, sizeof(Entry*));
  host_table.pool = (Entry*)malloc(ELEMENTS*sizeof(Entry));

  HANDLE_ERROR(cudaMemcpy(host_table.entries, dev_table.entries, dev_table.count*sizeof(Entry*), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(host_table.pool, dev_table.pool, ELEMENTS*sizeof(Entry), cudaMemcpyDeviceToHost));
  
  // Convert GPU addresses to CPU addresses
  for (int i=0; i<dev_table.count; i++) {
    if (host_table.entries[i] != NULL)
      host_table.entries[i] = (Entry*)((size_t)host_table.entries[i]-(size_t)dev_table.pool+(size_t)host_table.pool);
  }
  for (int i=0; i<ELEMENTS; i++) {
      if (host_table.pool[i].next != NULL)
          host_table.pool[i].next = (Entry*)((size_t)host_table.pool[i].next-(size_t)dev_table.pool+(size_t)host_table.pool);
  }
}

// copy table from GPU to CPU, then verify on CPU
void verify_table(const Table &dev_table) {
  Table table;
  copy_table_to_host(dev_table, table);

  // count: count the number of entries (in all buckets)
  int count = 0;
  for (size_t i=0; i<table.count; i++) {
    Entry *current = table.entries[i];
    while(current != NULL) {
      ++count; 
      if (hash(current->key, table.count) != i) {
        printf("%d hashed to %ld, but was located at %ld\n", current->key, hash(current->key, table.count), i);
      }
      current = current->next;
    }
  }
  if (count != ELEMENTS) {
    printf("%d elements found in hash table. Should be %ld\n", count, ELEMENTS);
  } else {
    printf("All %d elements found in hash table.\n", count);
  }
  // free CPU table 
  free(table.pool);
  free(table.entries);
}

// list of keys and list of values (each value = 1 pointer)
__global__ void add_to_table(unsigned int *keys, void **values, Table table, Lock *lock) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  while(tid<ELEMENTS) {
    unsigned int key = keys[tid];
    size_t hashValue = hash(key, table.count);
    // 1 warp = 32 threads going together in a clock step
    // only 1 thread in a warp can lock() at a time
    for (int i=0; i<32; i++) {
      // carry on the work ONLY FOR THREADS with WARP INDEX <i>
      if ((tid%32) == i) {
        // add to pool first
        Entry *location = &(table.pool[tid]);
        location->key = key;
        location->value = values[tid];
        // only one thread can modify <entries> at a time
        lock[hashValue].lock();
        location->next = table.entries[hashValue];
        table.entries[hashValue] = location;
        // NOTE: ensure that <location> fully written - avoid racing
        __threadfence();
        lock[hashValue].unlock();
      }
    }
    tid += stride;
  }
}

int main() {
  // allocating a big chunk of random ints
  unsigned int *buffer = (unsigned int*)big_random_block(SIZE);

  unsigned int *dev_keys;
  void **dev_values;
  HANDLE_ERROR(cudaMalloc((void**)&dev_keys, SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&dev_values, sizeof(void*)*ELEMENTS));
  // init dev_keys
  HANDLE_ERROR(cudaMemcpy(dev_keys, buffer, SIZE, cudaMemcpyHostToDevice));

  Table table;
  initialize_table(table, HASH_ENTRIES, ELEMENTS);

  // 1 lock for each bucket on table; on GPU
  Lock lock[HASH_ENTRIES];
  // memset(lock, 0, HASH_ENTRIES*sizeof(Lock));
  Lock *dev_lock;
  HANDLE_ERROR(cudaMalloc((void**)&dev_lock, HASH_ENTRIES*sizeof(Lock)));
  HANDLE_ERROR(cudaMemcpy(dev_lock, lock, HASH_ENTRIES*sizeof(Lock), cudaMemcpyHostToDevice));

  // timer
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  add_to_table<<<60,256>>>(dev_keys, dev_values, table, dev_lock);
  
  // timer
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time to hash: %3.1f ms\n", elapsedTime);

  verify_table(table);

  // clean
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  
  free_table(table);

  HANDLE_ERROR(cudaFree(dev_lock));
  HANDLE_ERROR(cudaFree(dev_keys));
  HANDLE_ERROR(cudaFree(dev_values));

  free(buffer);

  return 0;
}