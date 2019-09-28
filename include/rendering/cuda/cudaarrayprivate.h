#ifndef CUDAARRAYPRIVATE_H
#define CUDAARRAYPRIVATE_H

#include <stdint.h>

bool allocateManagedCudaMemory(void **data, const uint32_t size);
bool freeCudaMemory(void *data);

#endif // CUDAARRAYPRIVATE_H
