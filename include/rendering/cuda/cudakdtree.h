#ifndef CUDAKDTREE_H
#define CUDAKDTREE_H

#include <stdint.h>

#include <cxxutility/definitions.h>

#include "cuda/cudaarray.h"
#include "cuda/cudatypes.h"
#include "kdtreebuilder.h"

namespace ToyPT
{
namespace Rendering
{
namespace Cuda
{
namespace Types
{

struct Node
{
	Axis		axis			= Axis::X;
	Box			boundingBox;
	uint32_t	parentNodeIndex	= 0;
	uint32_t	leftNodeIndex	= 0;
	uint32_t	rightNodeIndex	= 0;
	uint32_t	leafBeginIndex	= 0;
	uint32_t	leafEndIndex	= 0;
};

using NodeBuffer	= Cuda::CudaArray<Node>;
using LeafBuffer	= Cuda::CudaArray<Cuda::Types::Triangle>;

}
}
}
}

#endif // CUDAKDTREE_H
