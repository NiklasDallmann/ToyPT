#include "rendering/cuda/cudaarrayprivate.h"

bool allocateManagedCudaMemory(void **data, const uint32_t size)
{
	bool returnValue = true;
	
	if (cudaMallocManaged(data, size) != cudaSuccess)
	{
		returnValue = false;
	}
	
	return returnValue;
}

bool freeCudaMemory(void *data)
{
	bool returnValue = true;
	
	if (cudaFree(data) != cudaSuccess)
	{
		returnValue = false;
	}
	
	return returnValue;
}
