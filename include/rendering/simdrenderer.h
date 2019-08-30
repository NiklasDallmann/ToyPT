#ifndef RENDERER_H
#define RENDERER_H

#include <array>
#include <immintrin.h>
#include <framebuffer.h>
#include <functional>
#include <stddef.h>
#include <vector>


#include <vector4.h>

#include "geometrycontainer.h"
#include "mesh.h"
#include "material.h"
#include "pointlight.h"
#include "randomnumbergenerator.h"
#include "ray.h"
#include "simdtypes.h"
#include "triangle.h"

namespace Rendering
{

class SimdRenderer
{	
public:
//	using CallBack = std::function<void(const uint32_t, const uint32_t)>;
	using CallBack = std::function<void()>;
	
	SimdRenderer();
	
	void render(FrameBuffer &frameBuffer, Obj::GeometryContainer &geometry, const CallBack &callBack, const bool &abort, const float fieldOfView = 75.0f,
				const uint32_t samples = 10, const uint32_t bounces = 2, const Math::Vector4 &skyColor = {});
	void renderAlbedoMap(FrameBuffer &frameBuffer, Obj::GeometryContainer &geometry, const float fieldOfView = 75.0f);
	void renderNormalMap(FrameBuffer &frameBuffer, Obj::GeometryContainer &geometry, const float fieldOfView = 75.0f);
	
private:
	enum class TraceType
	{
		Light,
		Object
	};
	
	struct IntersectionInfo
	{
		Simd::Mesh *mesh = nullptr;
		uint32_t triangleOffset = 0xFFFFFFFF;
		float u = 0.0f;
		float v = 0.0f;
	};
	
	struct Tile
	{
		uint32_t beginX = 0;
		uint32_t beginY = 0;
		
		uint32_t endX = 0;
		uint32_t endY = 0;
	};
	
	Simd::PreComputedTriangleBuffer _triangleBuffer;
	Simd::MeshBuffer _meshBuffer;
	
	static constexpr float _epsilon = 1.0E-7f;
	
	void _geometryToBuffer(const Obj::GeometryContainer &geometry, Simd::PreComputedTriangleBuffer &triangleBuffer, Simd::MeshBuffer &meshBuffer);
	
	__m256 _intersectSimd(const Ray &ray, Simd::PrecomputedTrianglePointer &data, __m256 &ts, __m256 &us, __m256 &v);
	
	template <TraceType T>
	float _traceRay(const Ray &ray, const Obj::GeometryContainer &geometry, IntersectionInfo &intersection);
	Math::Vector4 _castRay(const Ray &ray, const Obj::GeometryContainer &geometry, RandomNumberGenerator rng, const size_t maxBounces, const Math::Vector4 &skyColor);
	Math::Vector4 _castAlbedoRay(const Ray &ray, const Obj::GeometryContainer &geometry);
	Math::Vector4 _castNormalRay(const Ray &ray, const Obj::GeometryContainer &geometry);
	
	void _createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal);
	Math::Vector4 _createUniformHemisphere(const float r1, const float r2);
	Math::Vector4 _randomDirection(const Math::Vector4 &normal, RandomNumberGenerator &rng, float &cosinusTheta);
	
	Math::Vector4 _interpolateNormal(const Math::Vector4 &intersectionPoint, Simd::PrecomputedTrianglePointer &data);
	Math::Vector4 _brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v);
};

} // namespace Rendering

#endif // RENDERER_H
