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
	void render(FrameBuffer &frameBuffer, const float fieldOfView = 75, const size_t samples = 10, const size_t bounces = 2);
	
private:
	struct IntersectionInfo
	{
		Triangle *triangle = nullptr;
	};
	
	static constexpr float _epsilon = 0.000001;
	std::vector<Triangle> _triangles;
	std::vector<PointLight> _pointLights;
	
	bool _intersectTriangle(const float distance, const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle, const Math::Vector3D &normal);
	float _intersectPlane(const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle, const Math::Vector3D &normal);
	float _traceRay(const Math::Vector3D &direction, const Math::Vector3D &origin, IntersectionInfo &intersection);
	Math::Vector3D _castRay(const Math::Vector3D &direction, const Math::Vector3D &origin, const size_t samples = 10, const size_t bounce = 0,
							const size_t maxBounces = 4);
	void _createCoordinateSystem(const Math::Vector3D &N, Math::Vector3D &Nt, Math::Vector3D &Nb);
};

} // namespace Rendering

#endif // RENDERER_H
