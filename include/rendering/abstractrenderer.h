#ifndef ABSTRACTRENDERER_H
#define ABSTRACTRENDERER_H

#include <functional>
#include <stddef.h>
#include <vector>

#include <math/vector4.h>

#include "framebuffer.h"
#include "geometrycontainer.h"
#include "rendersettings.h"

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
	
	virtual void render(FrameBuffer &frameBuffer, const RenderSettings &settings, const Obj::GeometryContainer &geometry, const CallBack &callBack,
						const bool &abort) = 0;
};

} // namespace Rendering
} // namespace ToyPT

#endif // ABSTRACTRENDERER_H
