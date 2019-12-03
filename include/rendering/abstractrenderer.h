#ifndef ABSTRACTRENDERER_H
#define ABSTRACTRENDERER_H

#include <functional>
#include <stddef.h>
#include <vector>

#include <math/vector4.h>

#include "framebuffer.h"
#include "geometrycontainer.h"

namespace ToyPT
{
namespace Rendering
{

class AbstractRenderer
{
public:
	using CallBack = std::function<void(const uint32_t x0, const uint32_t y0, const uint32_t x1, const uint32_t y1)>;
	
	AbstractRenderer() = default;
	virtual ~AbstractRenderer() = default;
	
	virtual void render(FrameBuffer &frameBuffer, const Obj::GeometryContainer &geometry, const Obj::GeometryContainer &lights, const CallBack &callBack,
						const bool &abort, const float fieldOfView = 75.0f, const uint32_t samples = 10, const uint32_t bounces = 2,
						const uint32_t tileSize = 32, const Math::Vector4 &skyColor = {}) = 0;
	
protected:
	struct Tile
	{
		uint32_t beginX = 0;
		uint32_t beginY = 0;
		
		uint32_t endX = 0;
		uint32_t endY = 0;
	};
};

} // namespace Rendering
} // namespace ToyPT

#endif // ABSTRACTRENDERER_H
