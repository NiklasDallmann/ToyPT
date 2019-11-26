#ifndef OPENCLRENDERER_H
#define OPENCLRENDERER_H

#include <vector_types.h>
#include <math/vector4.h>

#include "abstractrenderer.h"
#include "cudaarray.h"
#include "cudatypes.h"
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
	
	virtual void render(FrameBuffer &frameBuffer, const Obj::GeometryContainer &geometry, const Obj::GeometryContainer &lights, const CallBack &callBack,
						const bool &abort, const float fieldOfView = 75.0f, const uint32_t samples = 10, const uint32_t bounces = 2,
						const uint32_t tileSize = 32, const Math::Vector4 &skyColor = {});
	
private:
	void _geometryToBuffer(const Obj::GeometryContainer &geometry, CudaArray<Cuda::Types::Triangle> &triangleBuffer,
						   CudaArray<Cuda::Types::Mesh> &meshBuffer, CudaArray<Material> &materialBuffer);
};

} // namespace ToyPT::Rendering::Cuda

#endif // OPENCLRENDERER_H
