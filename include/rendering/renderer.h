#ifndef RENDERER_H
#define RENDERER_H

#include <array>
#include <emmintrin.h>
#include <immintrin.h>
#include <framebuffer.h>
#include <stddef.h>
#include <vector>
#include <vector4.h>

#include "mesh.h"
#include "material.h"
#include "pointlight.h"
#include "triangle.h"

namespace Rendering
{

class Renderer
{
public:
	Renderer();
	
	std::vector<Vertex> vertexBuffer;
	std::vector<Math::Vector4> uvBuffer;
	std::vector<Math::Vector4> normalBuffer;
	std::vector<Triangle> triangleBuffer;
	std::vector<Material> materialBuffer;
	std::vector<Mesh> meshBuffer;
	
	void render(FrameBuffer &frameBuffer, const float fieldOfView = 75.0f, const size_t samples = 10, const size_t bounces = 2);
	
private:
	struct IntersectionInfo
	{
		Mesh *mesh = nullptr;
		Triangle *triangle = nullptr;
		float u = 0;
		float v = 0;
	};
	
	struct CoordinateBuffer
	{
		std::vector<float> x;
		std::vector<float> y;
		std::vector<float> z;
	};
	
	struct Ray
	{
		Math::Vector4 origin;
		Math::Vector4 direction;
	};
	
	static constexpr float _epsilon = 1.0E-7f;
	
	bool _intersectMoellerTrumbore(const Ray &ray, const Triangle *triangle, float &t, float &u, float &v);
	
//	__m256 _intersectAvx2(const Ray)
	
	float _traceRay(const Ray &ray, IntersectionInfo &intersection);
	Math::Vector4 _castRay(const Ray &ray,
							const size_t maxBounces = 4, const bool debug = false);
	
	void _createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal);
	Math::Vector4 _createUniformHemisphere(const float r1, const float r2);
	
	Math::Vector4 _interpolateNormal(const IntersectionInfo &intersection, const Math::Vector4 &intersectionPoint);
	Math::Vector4 _brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v);
};

} // namespace Rendering

#endif // RENDERER_H
