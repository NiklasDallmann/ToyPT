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
#include "rendersettings.h"
#include "storage.h"
#include "triangle.h"

namespace ToyPT::Rendering
{

class SimdRenderer : public AbstractRenderer
{	
public:
	SimdRenderer();
	
	virtual void render(
		FrameBuffer						&frameBuffer,
		const RenderSettings			&settings,
		const Obj::GeometryContainer	&geometry,
		const CallBack					&callBack,
		const bool						&abort) override;
	
private:
	struct IntersectionInfo
	{
		Storage::Mesh	*mesh = nullptr;
		uint32_t		triangleOffset = 0xFFFFFFFF;
		float			u = 0.0f;
		float			v = 0.0f;
	};
	
	Storage::PreComputedTriangleBuffer	_objectTriangleBuffer;
	Storage::MeshBuffer					_objectMeshBuffer;
	
	Storage::PreComputedTriangleBuffer	_lightTriangleBuffer;
	Storage::MeshBuffer					_lightMeshBuffer;
	
	void _geometryToBuffer(
		const Obj::GeometryContainer		&geometry, 
		Storage::PreComputedTriangleBuffer	&triangleBuffer, 
		Storage::MeshBuffer					&meshBuffer);
	
	inline __m256 _intersectSimd(
		const Ray							&ray,
		Storage::PrecomputedTrianglePointer	&data,
		__m256								&ts,
		__m256								&us,
		__m256								&vs,
		__m256i								&meshOffsets);
	
	inline float _traceRay(const Ray						&ray,
		IntersectionInfo				&intersection);
	
	inline Math::Vector4 _castRay(
		const Ray						&ray,
		const Obj::GeometryContainer	&geometry,
		RandomNumberGenerator			rng,
		const size_t					maxBounces,
		const Math::Vector4				&skyColor);
	
	inline void _createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal);
	inline Math::Vector4 _createUniformHemisphere(const float r1, const float r2);
	inline Math::Vector4 _randomDirection(const Math::Vector4 &normal, RandomNumberGenerator &rng, float &cosinusTheta);
	
	inline Math::Vector4 _interpolateNormal(const Math::Vector4 &intersectionPoint, Storage::PrecomputedTrianglePointer &data);
};

} // namespace ToyPT::Rendering

#endif // SIMDRENDERER_H
