#ifndef RENDERER_H
#define RENDERER_H

#include <array>
#include <framebuffer.h>
#include <stddef.h>
#include <vector>
#include <vector3d.h>

#include "pointlight.h"
#include "triangle.h"

namespace Rendering
{

class Renderer
{
public:
	Renderer();
	
	void setTriangles(const std::vector<Triangle> &triangles);
	void setPointLights(const std::vector<PointLight> &pointLights);
	void render(FrameBuffer &frameBuffer, double fieldOfView = 75);
	
private:
	static constexpr double _epsilon = 0.000001;
	std::vector<Triangle> _triangles;
	std::vector<PointLight> _pointLights;
	
	bool _intersectTriangle(const double distance, const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle);
	double _intersectPlane(const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle);
	double _traceRay(const Math::Vector3D &direction, const Math::Vector3D &origin, Triangle **triangle);
	Math::Vector3D _castRay(const Math::Vector3D &direction, const Math::Vector3D &origin, const size_t samples = 1, const size_t bounce = 0, const size_t maxBounces = 4);
	Math::Vector3D _randomDirectionInHemisphere(const Math::Vector3D &normal);
};

} // namespace Rendering

#endif // RENDERER_H
