#ifndef SIMDRENDERER_H
#define SIMDRENDERER_H

#include <array>
#include <immintrin.h>
#include <framebuffer.h>
#include <functional>
#include <stddef.h>
#include <vector>

#include <math/vector4.h>

#include "abstractrenderer.h"
#include "geometrycontainer.h"
#include "mesh.h"
#include "material.h"
#include "pointlight.h"
#include "randomnumbergenerator.h"
#include "ray.h"
#include "storage.h"
#include "triangle.h"

namespace Rendering
{

class SimdRenderer : public AbstractRenderer
{	
public:
	SimdRenderer();
	
	virtual void render(FrameBuffer &frameBuffer, const Obj::GeometryContainer &geometry, const Obj::GeometryContainer &lights, const CallBack &callBack,
						const bool &abort, const float fieldOfView = 75.0f, const uint32_t samples = 10, const uint32_t bounces = 2,
						const uint32_t tileSize = 32, const Math::Vector4 &skyColor = {}) override;
	
private:
	struct IntersectionInfo
	{
		Storage::Mesh *mesh = nullptr;
		uint32_t triangleOffset = 0xFFFFFFFF;
		float u = 0.0f;
		float v = 0.0f;
	};
	
	Storage::PreComputedTriangleBuffer _objectTriangleBuffer;
	Storage::MeshBuffer _objectMeshBuffer;
	
	Storage::PreComputedTriangleBuffer _lightTriangleBuffer;
	Storage::MeshBuffer _lightMeshBuffer;
	
	void _geometryToBuffer(const Obj::GeometryContainer &geometry, Storage::PreComputedTriangleBuffer &triangleBuffer, Storage::MeshBuffer &meshBuffer);
	
	__m256 _intersectSimd(const Ray &ray, Storage::PrecomputedTrianglePointer &data, __m256 &ts, __m256 &us, __m256 &v);
	
	float _traceRay(const Ray &ray, const Obj::GeometryContainer &geometry, IntersectionInfo &intersection);
	Math::Vector4 _castRay(const Ray &ray, const Obj::GeometryContainer &geometry, const Obj::GeometryContainer &lights, RandomNumberGenerator rng,
						   const size_t maxBounces, const Math::Vector4 &skyColor);
	
	void _createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal);
	Math::Vector4 _createUniformHemisphere(const float r1, const float r2);
	Math::Vector4 _randomDirection(const Math::Vector4 &normal, RandomNumberGenerator &rng, float &cosinusTheta);
	
	Math::Vector4 _interpolateNormal(const Math::Vector4 &intersectionPoint, Storage::PrecomputedTrianglePointer &data);
	float _ggxChi(const float x);
	float _ggxPartial(const Math::Vector4 &v, const Math::Vector4 &h, const Math::Vector4 &n, const float a_2);
	Math::Vector4 _brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v, const float cosinusTheta);
};

} // namespace Rendering

#endif // SIMDRENDERER_H
