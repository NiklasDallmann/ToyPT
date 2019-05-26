#ifndef RENDERER_H
#define RENDERER_H

#include <array>
#include <framebuffer.h>
#include <stddef.h>
#include <vector3d.h>

namespace Rendering
{

class Renderer
{
public:
	using Triangle = std::array<Math::Vector3D, 3>;
	
	Renderer();
	
	void render(FrameBuffer &frameBuffer, double fieldOfView = 75);
};

} // namespace Rendering

#endif // RENDERER_H
