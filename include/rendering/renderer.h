#ifndef RENDERER_H
#define RENDERER_H

#include <array>
#include <immintrin.h>
#include <framebuffer.h>
#include <stddef.h>
#include <vector>


#include <vector4.h>

#include "geometrycontainer.h"
#include "mesh.h"
#include "material.h"
#include "pointlight.h"
#include "ray.h"
#include "simdtypes.h"
#include "triangle.h"

namespace Rendering
{

class Renderer
{
public:
	Renderer();
	
	Obj::GeometryContainer geometry;
	
	void render(FrameBuffer &frameBuffer, const float fieldOfView = 75.0f, const size_t samples = 10, const size_t bounces = 2);
	
private:
	struct IntersectionInfo
	{
		Obj::Mesh *mesh = nullptr;
		Obj::Triangle *triangle = nullptr;
		float u = 0;
		float v = 0;
	};
	
//	struct IntersectionInfo
//	{
//		Simd::Mesh *mesh = nullptr;
//		uint32_t triangleOffset = 0;
//		float u = 0;
//		float v = 0;
//	};
	
	Simd::VertexBuffer _vertexBuffer;
	Simd::NormalBuffer _normalBuffer;
	Simd::MeshBuffer _meshBuffer;
	
	static constexpr float _epsilon = 1.0E-7f;
	static constexpr uint32_t _avx2FloatCount = sizeof (__m256) / sizeof (float);
	
	void _geometryToBuffer(const Obj::GeometryContainer &geometry, Simd::VertexBuffer &vertexBuffer, Simd::NormalBuffer &normalBuffer, Simd::MeshBuffer &meshBuffer);
	
	bool _intersectMoellerTrumbore(const Ray &ray, const Obj::Triangle *triangle, float &t, float &u, float &v);
	
	__m256i _intersectAvx2(const Ray &ray, float *verticesXs, float *verticesYs, float *verticesZs, __m256 &ts, __m256 &us, __m256 &vs);
	
	float _traceRay(const Ray &ray, IntersectionInfo &intersection);
	Math::Vector4 _castRay(const Ray &ray, const size_t maxBounces = 4, const bool debug = false);
	
	void _createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal);
	Math::Vector4 _createUniformHemisphere(const float r1, const float r2);
	
	Math::Vector4 _interpolateNormal(const IntersectionInfo &intersection, const Math::Vector4 &intersectionPoint);
	Math::Vector4 _brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v);
};

} // namespace Rendering

#endif // RENDERER_H
