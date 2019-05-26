#ifndef RENDERER_H
#define RENDERER_H

#include <array>
#include <framebuffer.h>
#include <stddef.h>
#include <vector>
#include <vector3d.h>

#include "triangle.h"

namespace Rendering
{

class Renderer
{
public:
	Renderer();
	
	void setTriangles(const std::vector<Triangle> &triangles);
	void render(FrameBuffer &frameBuffer, double fieldOfView = 75);
	
private:
	static constexpr double _epsilon = 0.0001;
	std::vector<Triangle> _triangles;
	
	bool _intersectTriangle(const double distance, const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle);
	double _intersectPlane(const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle);
	Math::Vector3D _castRay(const Math::Vector3D &direction, const Math::Vector3D &origin);
};

} // namespace Rendering

#endif // RENDERER_H
