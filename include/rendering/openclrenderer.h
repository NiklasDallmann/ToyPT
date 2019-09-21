#ifndef OPENCLRENDERER_H
#define OPENCLRENDERER_H

#include <CL/cl.hpp>

#include <math/vector4.h>

#include "abstractrenderer.h"
#include "framebuffer.h"
#include "geometrycontainer.h"
#include "storage.h"

namespace Rendering
{

class OpenCLRenderer : public AbstractRenderer
{
public:
	OpenCLRenderer();
	
	virtual void render(FrameBuffer &frameBuffer, Obj::GeometryContainer &geometry, const CallBack &callBack, const bool &abort, const float fieldOfView = 75.0f,
				const uint32_t samples = 10, const uint32_t bounces = 2, const uint32_t tileSize = 32, const Math::Vector4 &skyColor = {}) override;
	
private:
	cl::Platform _platform;
	cl::Device _device;
	cl::Context _context;
	cl::Program _program;
	
	Storage::PreComputedTriangleBuffer _triangleBuffer;
	Storage::MeshBuffer _meshBuffer;
	
	void _initializeHardware();
	bool _buildKernel();
};

} // namespace Rendering

#endif // OPENCLRENDERER_H
