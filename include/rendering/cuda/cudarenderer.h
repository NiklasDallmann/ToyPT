#ifndef OPENCLRENDERER_H
#define OPENCLRENDERER_H

#include <vector_types.h>
#include <math/vector4.h>

#include "abstractrenderer.h"
#include "cudaarray.h"
#include "cudatypes.h"
#include "cuda/cudakdtree.h"
#include "framebuffer.h"
#include "geometrycontainer.h"
#include "material.h"
#include "storage.h"

namespace ToyPT::Rendering::Cuda
{

class CudaRenderer : public AbstractRenderer
{
public:
	CudaRenderer() = default;
	
	virtual void render(
		FrameBuffer						&frameBuffer,
		const RenderSettings			&settings,
		const Obj::GeometryContainer	&geometry,
		const CallBack					&callBack,
		const bool						&abort);
	
private:
	void		_geometryToBuffer(
					const Obj::GeometryContainer		&geometry,
					CudaArray<Cuda::Types::Triangle>	&triangleBuffer,
					CudaArray<Cuda::Types::Mesh>		&meshBuffer,
					CudaArray<Material>					&materialBuffer);
	
	void		_buildKdTree(const Obj::GeometryContainer		&geometry, CudaArray<Types::Node> &nodeBuffer,
					CudaArray<Cuda::Types::Triangle>	&triangleBuffer,
					CudaArray<Cuda::Types::Mesh>		&meshBuffer,
					CudaArray<Material>					&materialBuffer);
	
	void		_traverseKdTree(
					const Obj::GeometryContainer		&geometry,
					const Node							*node,
					std::vector<Cuda::Types::Node>		&deviceNodes,
					std::vector<Cuda::Types::Triangle>	&deviceTriangles);
};

} // namespace ToyPT::Rendering::Cuda

#endif // OPENCLRENDERER_H
