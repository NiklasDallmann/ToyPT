#ifndef RENDERER_H
#define RENDERER_H

#include <array>
#include <framebuffer.h>
#include <stddef.h>
#include <vector>
#include <vector4.h>

#include "abstractmesh.h"
#include "pointlight.h"
#include "triangle.h"

namespace Rendering
{

class Renderer
{
public:
	Renderer();
	
	void setMeshes(const std::vector<AbstractMesh *> &meshes);
	void setPointLights(const std::vector<PointLight> &pointLights);
	void render(FrameBuffer &frameBuffer, const float fieldOfView = 75.0f, const size_t samples = 10, const size_t bounces = 2);
	
private:
	struct IntersectionInfo
	{
		AbstractMesh *mesh = nullptr;
		Triangle *triangle = nullptr;
	};
	
	static constexpr float _epsilon = 0.000001f;
	std::vector<AbstractMesh *> _meshes;
	std::vector<PointLight> _pointLights;
	
	bool _intersectTriangle(const float distance, const Math::Vector4 &direction, const Math::Vector4 &origin, const Triangle &triangle, const Math::Vector4 &normal);
	float _intersectPlane(const Math::Vector4 &direction, const Math::Vector4 &origin, const Triangle &triangle, const Math::Vector4 &normal);
	float _traceRay(const Math::Vector4 &direction, const Math::Vector4 &origin, IntersectionInfo &intersection);
	Math::Vector4 _castRay(const Math::Vector4 &direction, const Math::Vector4 &origin, const size_t bounce = 0,
							const size_t maxBounces = 4);
	void _createCoordinateSystem(const Math::Vector4 &N, Math::Vector4 &Nt, Math::Vector4 &Nb);
};

} // namespace Rendering

#endif // RENDERER_H
