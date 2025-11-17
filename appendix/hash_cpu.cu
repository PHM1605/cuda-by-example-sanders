#include "../common/book.h"
#include "lock.h"

#define HASH_ENTRIES 1024
#define SIZE (100*1024*1024) // total size of all random numbers generated
#define ELEMENTS (SIZE/sizeof(unsigned int)) // number of random numbers generated

struct Entry {
  unsigned int key;
  void* value;
  Entry *next;
};

struct Table {
  size_t count; // number of buckets
  Entry **entries; // pointer to a list of pointers (to Entry)
  Entry *pool; // all Entries
  Entry *firstFree; // last Entry
};

void initialize_table(Table& table, int entries, int elements) {
  table.count = entries;
  table.entries = (Entry**)calloc(entries, sizeof(Entry*));
  table.pool = (Entry*)malloc(elements*sizeof(Entry));
  table.firstFree = table.pool;
}

void free_table(Table& table) {
  free(table.entries);
  free(table.pool);
}

size_t hash(unsigned int key, size_t count) {
  return key % count;
}

void add_to_table(Table &table, unsigned int key, void* value) {
  // which bucket that new Entry belongs to
  size_t hashValue = hash(key, table.count);
  
  // add that new Entry to pool
  Entry *location = table.firstFree++;
  location->key = key;
  location->value = value; 
  
  // add that new Entry to the START of Bucket <key> in Table
  location->next = table.entries[hashValue];
  table.entries[hashValue] = location;
}

void verify_table(const Table &table) {
  
}

int main() {
  // allocating a big chunk of random ints
  unsigned int *buffer = (unsigned int*)big_random_block(SIZE);

  return 0;
}