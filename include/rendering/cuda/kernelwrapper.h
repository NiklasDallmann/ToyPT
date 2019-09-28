#ifndef KERNELWRAPPER_H
#define KERNELWRAPPER_H

#include <stdint.h>

#include "rendering/cuda/cudaarray.h"

struct KernelWrapper
{
	static void workItemDistribution(const uint32_t workItems, uint32_t &blocks, uint32_t &threads);
};

#endif // KERNELWRAPPER_H
