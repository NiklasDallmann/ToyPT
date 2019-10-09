#include <cmath>

#include "debugstream.h"
#include "rendering/cuda/kernelwrapper.h"

namespace ToyPT
{
namespace Rendering
{
namespace Cuda
{

void KernelWrapper::workItemDistribution(const uint32_t workItems, uint32_t &blocks, uint32_t &threads)
{
	threads = std::min(256u, workItems);
	blocks = std::max(1u, (workItems / threads));
}

}
}
}
