#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <math/matrix4x4.h>
#include <math/simdvector3pack.h>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <iomanip>
#include <iostream>
#include <sstream>

#include "debugstream.h"
#include "simd/simdrenderer.h"

namespace Rendering
{

SimdRenderer::SimdRenderer() :
	AbstractRenderer()
{
}

void SimdRenderer::render(FrameBuffer &frameBuffer, const Obj::GeometryContainer &geometry, const Obj::GeometryContainer &lights, const CallBack &callBack,
						  const bool &abort, const float fieldOfView, const uint32_t samples, const uint32_t bounces, const uint32_t tileSize,
						  const Math::Vector4 &skyColor)
{
	const uint32_t width = frameBuffer.width();
	const uint32_t height = frameBuffer.height();
	float fovRadians = fieldOfView / 180.0f * float(M_PI);
	float zCoordinate = -(width/(2.0f * std::tan(fovRadians / 2.0f)));
	std::stringstream stream;
	
	this->_geometryToBuffer(geometry, this->_objectTriangleBuffer, this->_objectMeshBuffer);
	
	std::random_device device;
	const uint32_t tilesVertical = height / tileSize + ((height % tileSize) > 0);
	const uint32_t tilesHorizontal = width / tileSize + ((width % tileSize) > 0);
	
#pragma omp parallel for schedule(dynamic, 1) collapse(2)
	for (uint32_t tileVertical = 0; tileVertical < tilesVertical; tileVertical++)
	{
		for (uint32_t tileHorizontal = 0; tileHorizontal < tilesHorizontal; tileHorizontal++)
		{
			uint32_t startVertical = tileSize * tileVertical;
			uint32_t startHorizontal = tileSize * tileHorizontal;
			uint32_t endVertical = std::min(startVertical + tileSize, height);
			uint32_t endHorizontal = std::min(startHorizontal + tileSize, width);
			
			for (uint32_t h = startVertical; h < endVertical; h++)
			{
				for (uint32_t w = startHorizontal; (w < endHorizontal) & !abort; w++)
				{
					RandomNumberGenerator rng(device());
					Math::Vector4 color;
					
					for (size_t sample = 1; sample <= samples; sample++)
					{
						float offsetX, offsetY;
						const float scalingFactor = 1.0f / float(std::numeric_limits<uint32_t>::max());
						offsetX = rng.get(scalingFactor)  - 0.5f;
						offsetY = rng.get(scalingFactor) - 0.5f;
						
						float x = (w + offsetX + 0.5f) - (width / 2.0f);
						float y = -(h + offsetY + 0.5f) + (height / 2.0f);
						
						Math::Vector4 direction{x, y, zCoordinate};
						direction.normalize();
					
						color += this->_castRay({{0, 0, 0}, direction}, geometry, lights, rng, bounces, skyColor);
					}
					
					frameBuffer.setPixel(w, h, (color / float(samples)));
				}
			}
			
			if (!abort)
			{
				callBack(startHorizontal, startVertical, endHorizontal, endVertical);
			}
		}
	}
}

void SimdRenderer::_geometryToBuffer(const Obj::GeometryContainer &geometry, Storage::PreComputedTriangleBuffer &triangleBuffer, Storage::MeshBuffer &meshBuffer)
{
	Storage::geometryToBuffer(geometry, triangleBuffer, meshBuffer);
	
	// Apply padding to a multiple of the used vector size
	const uint32_t triangleBufferSize = uint32_t(triangleBuffer.size());
	const uint32_t paddingTriangles = Storage::avx2FloatCount - (triangleBufferSize % Storage::avx2FloatCount);
	
	for (uint32_t i = 0; i < paddingTriangles; i++)
	{
		triangleBuffer.append({}, {}, {}, {}, {}, {}, Storage::maskFalse);
	}
}

__m256 SimdRenderer::_intersectSimd(const Ray &ray, Storage::PrecomputedTrianglePointer &data, __m256 &ts, __m256 &us, __m256 &vs)
{
	__m256 returnValue;
	__m256 determinant, inverseDeterminant, epsilon;
	
	Math::SimdVector3Pack origin, direction, v0, v1, v2, e01, e02, v0o, pVector, qVector;
	
	// Duplicate ray data
	origin.x = _mm256_set1_ps(ray.origin.x());
	origin.y = _mm256_set1_ps(ray.origin.y());
	origin.z = _mm256_set1_ps(ray.origin.z());
	
	direction.x = _mm256_set1_ps(ray.direction.x());
	direction.y = _mm256_set1_ps(ray.direction.y());
	direction.z = _mm256_set1_ps(ray.direction.z());
	
	epsilon = _mm256_set1_ps(Math::epsilon);
	
	// Load vertex coordinates
	v0.loadUnaligned(data.v0.x, data.v0.y, data.v0.z);
	v1.loadUnaligned(data.v1.x, data.v1.y, data.v1.z);
	v2.loadUnaligned(data.v2.x, data.v2.y, data.v2.z);
	e01.loadUnaligned(data.e01.x, data.e01.y, data.e01.z);
	e02.loadUnaligned(data.e02.x, data.e02.y, data.e02.z);
	
	data += Storage::avx2FloatCount;
	
	// Sub
	v0o = origin - v0;
	
	// Cross
	pVector = direction.crossProduct(e02);
	qVector = v0o.crossProduct(e01);
	
	determinant = e01.dotProduct(pVector);
	inverseDeterminant = _mm256_div_ps(_mm256_set1_ps(1.0f), determinant);
	
	us = v0o.dotProduct(pVector) * inverseDeterminant;
	vs = direction.dotProduct(qVector) * inverseDeterminant;
	ts = e02.dotProduct(qVector) * inverseDeterminant;
	
	// Conditions
	__m256 c0 = _mm256_cmp_ps(determinant, epsilon, _CMP_LT_OQ);
	__m256 c1 = _mm256_cmp_ps(us, _mm256_set1_ps(0.0f), _CMP_LT_OQ);
	__m256 c2 = _mm256_cmp_ps(us, _mm256_set1_ps(1.0f), _CMP_GT_OQ);
	__m256 c3 = _mm256_cmp_ps(vs, _mm256_set1_ps(0.0f), _CMP_LT_OQ);
	__m256 c4 = _mm256_cmp_ps(_mm256_add_ps(us, vs), _mm256_set1_ps(1.0f), _CMP_GT_OQ);
	__m256 c5 = _mm256_cmp_ps(ts, epsilon, _CMP_GT_OQ);
	__m256 c = _mm256_or_ps(c0, _mm256_or_ps(c1, _mm256_or_ps(c2, _mm256_or_ps(c3, c4))));
	
	// Convert to integer vector of values of either 0x00 or 0x01
	returnValue = _mm256_and_ps(_mm256_andnot_ps(c, _mm256_set1_ps(0x01)), c5);
	
	return returnValue;
}

float SimdRenderer::_traceRay(const Ray &ray, const Obj::GeometryContainer &geometry, IntersectionInfo &intersection)
{
	float returnValue = 0.0f;
	
	uint32_t triangleCount = this->_objectTriangleBuffer.size();
	bool intersectionFound = false;
	Storage::Mesh *nearestMesh = nullptr;
	uint32_t nearestTriangle = 0xFFFFFFFF;
	float newDistance = 0.0f;
	float distance = std::numeric_limits<float>::max();
	float u = 0;
	float v = 0;
	newDistance = distance;
	
	// Intersect triangles
	Storage::PrecomputedTrianglePointer dataPointer = this->_objectTriangleBuffer.data();
	
	uint32_t triangleIndex = 0;
	uint32_t avx2Loops = triangleCount - (triangleCount % Storage::avx2FloatCount);
	
	for (; triangleIndex < avx2Loops; triangleIndex += Storage::avx2FloatCount)
	{
		__m256 ts, us, vs;
		__m256 intersected = this->_intersectSimd(ray, dataPointer, ts, us, vs);
		
		for (uint32_t index = 0; index < Storage::avx2FloatCount; index++)
		{
			newDistance = ts[index];
			if ((newDistance < distance) & bool(intersected[index]))
			{
				intersectionFound = true;
				distance = newDistance;
				nearestTriangle = triangleIndex + index;
			}
		}
	}
	
	// Find corresponding mesh
	if (intersectionFound)
	{
		for (uint32_t meshIndex = 0; meshIndex < this->_objectMeshBuffer.size(); meshIndex++)
		{
			Storage::Mesh &mesh = this->_objectMeshBuffer[meshIndex];
			
			if ((nearestTriangle >= mesh.triangleOffset) &
				(nearestTriangle < (mesh.triangleOffset + mesh.triangleCount)))
			{
				nearestMesh = &mesh;
				break;
			}
		}
	}
	
	returnValue = distance;
	intersection.mesh = nearestMesh;
	intersection.triangleOffset = nearestTriangle;
	intersection.u = u;
	intersection.v = v;
	
	return returnValue;
}

Math::Vector4 SimdRenderer::_castRay(const Ray &ray, const Obj::GeometryContainer &geometry, const Obj::GeometryContainer &lights, RandomNumberGenerator rng, const size_t maxBounces, const Math::Vector4 &skyColor)
{
	Math::Vector4 returnValue = {0.0f, 0.0f, 0.0f};
	Math::Vector4 mask = {1.0f, 1.0f, 1.0f};
	
	Math::Vector4 currentDirection = ray.direction;
	Math::Vector4 currentOrigin = ray.origin;
	
	float cosinusTheta;
	
	for (size_t currentBounce = 0; currentBounce < maxBounces; currentBounce++)
	{
		Math::Vector4 intersectionPoint;
		IntersectionInfo objectIntersection, lightIntersection;
		Math::Vector4 normal;
		float objectDistance = this->_traceRay({currentOrigin, currentDirection}, geometry, objectIntersection);
		
		intersectionPoint = currentOrigin + (objectDistance * currentDirection);
		
		if (objectIntersection.mesh != nullptr)
		{
			Material objectMaterial = geometry.materialBuffer[objectIntersection.mesh->materialOffset];
			Math::Vector4 objectColor = objectMaterial.color;
//			Material lightMaterial;
//			Math::Vector4 lightColor;
//			Math::Vector4 directLight;
//			float lightEmittance = 0.0f;
			
			// Calculate normal
			Storage::PrecomputedTrianglePointer dataPointer = this->_objectTriangleBuffer.data() + objectIntersection.triangleOffset;
			normal = this->_interpolateNormal(intersectionPoint, dataPointer);
			
			// Calculate new origin and offset
			currentOrigin = intersectionPoint + (Math::epsilon * normal);
			
			// Direct illumination
//			float lightCosinusTheta;
//			Math::Vector4 lightDirection = this->_randomDirection(normal, rng, lightCosinusTheta);
//			float lightDistance = this->_traceRay({currentOrigin, lightDirection}, lights, lightIntersection);
			
//			if (lightIntersection.mesh != nullptr)
//			{
//				lightMaterial = geometry.materialBuffer[lightIntersection.mesh->materialOffset];
//				lightColor = lightMaterial.color();
//				lightEmittance = lightMaterial.emittance();
//				directLight = lightColor * lightEmittance;
//			}
			
			// Global illumination
			Math::Vector4 newDirection, reflectedDirection, diffuseDirection;
			Math::Vector4 diffuse, specular;
			
			diffuseDirection = this->_randomDirection(normal, rng, cosinusTheta);
			reflectedDirection = (currentDirection - 2.0f * currentDirection.dotProduct(normal) * normal).normalize();
			
			newDirection = Math::lerp(diffuseDirection, reflectedDirection, objectMaterial.roughness);
//			newDirection = diffuseDirection;
			
			// Cook-Torrance
//			specular = this->_brdf(objectMaterial, normal, newDirection, -currentDirection, cosinusTheta);
			// Lambert
//			const float pdf = 2.0f;
			specular = Math::Vector4{1.0f, 1.0f, 1.0f} * (1.0f - objectMaterial.roughness);
			diffuse = Math::Vector4{1.0f, 1.0f, 1.0f} - specular;
			
			currentDirection = newDirection;
			
			returnValue += objectMaterial.emittance * mask;
//			returnValue += directLight * mask;
//			mask *= (objectColor * diffuse + specular) * cosinusTheta;
			mask *= (2.0f * objectColor * diffuse + specular) * cosinusTheta;
//			mask *= 2.0f * objectColor * cosinusTheta;
//			returnValue = specular;
		}
		else
		{
			returnValue += skyColor * mask;
			break;
		}
	}
	
	return returnValue;
}

void SimdRenderer::_createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal)
{
	const Math::Vector4 a = Math::Vector4{normal.z(), 0.0f, -normal.x()};
	const Math::Vector4 b = Math::Vector4{0.0f, -normal.z(), normal.y()};
	float t = std::abs(normal.x()) > std::abs(normal.y());
	
	tangentNormal = Math::lerp(a, b, t).normalize();
	
	binormal = normal.crossProduct(tangentNormal);
}

Math::Vector4 SimdRenderer::_createUniformHemisphere(const float r1, const float r2)
{
	float sinTheta = std::sqrt(1.0f - r1 * r1);
	float phi = 2.0f * float(M_PI) * r2;
	float x = sinTheta * std::cos(phi);
	float z = sinTheta * std::sin(phi);
	return {x, r1, z};
}

Math::Vector4 SimdRenderer::_randomDirection(const Math::Vector4 &normal, RandomNumberGenerator &rng, float &cosinusTheta)
{
	float ratio;
	
	Math::Vector4 Nt;
	Math::Vector4 Nb;
	
	this->_createCoordinateSystem(normal, Nt, Nb);
	
	// Generate hemisphere
	constexpr float scalingFactor = 1.0f / float(std::numeric_limits<uint32_t>::max());
	cosinusTheta = std::pow(rng.get(scalingFactor), 0.5f);
	ratio = rng.get(scalingFactor);
	
	Math::Vector4 sample = this->_createUniformHemisphere(cosinusTheta, ratio);
	
	Math::Matrix4x4 localToWorldMatrix{
		{Nb.x(), normal.x(), Nt.x()},
		{Nb.y(), normal.y(), Nt.y()},
		{Nb.z(), normal.z(), Nt.z()}
	};
	
	return (localToWorldMatrix * sample).normalize();
}

Math::Vector4 SimdRenderer::_interpolateNormal(const Math::Vector4 &intersectionPoint, Storage::PrecomputedTrianglePointer &data)
{
	Math::Vector4 returnValue, p, n0, n1, n2, n01, n02, v0, v1, v2, e01, e02, v12, v0p, v1p, v2p, vab, v2ab;
	
	v0 = {*data.v0.x, *data.v0.y, *data.v0.z};
	v1 = {*data.v1.x, *data.v1.y, *data.v1.z};
	v2 = {*data.v2.x, *data.v2.y, *data.v2.z};
	e01 = {*data.e01.x, *data.e01.y, *data.e01.z};
	e02 = {*data.e02.x, *data.e02.y, *data.e02.z};
	n0 = {*data.n0.x, *data.n0.y, *data.n0.z};
	n1 = {*data.n1.x, *data.n1.y, *data.n1.z};
	n2 = {*data.n2.x, *data.n2.y, *data.n2.z};
	
	data++;
	
	p = intersectionPoint;
	v12 = v2 - v1;
	v0p = p - v0;
	v1p = p - v1;
	v2p = p - v2;
	
	float a, b, denominator;
	
	denominator = (e01.x() * v2p.y() - v2p.x() * e01.y()) + Math::epsilon;
	a = (-(v0.x() * v2p.y() - v2p.x() * v0.y() + v2p.x() * v2.y() - v2.x() * v2p.y())) / denominator;
//	b = (e01.x() * v0.y() - e01.x() * v2.y() - v0.x() * e01.y() + v2.x() * e01.y()) / denominator;
	
	vab = v0 + a * e01;
	
	n01 = Math::lerp(n1, n0, a).normalize();
	v2ab = vab - v2;
	
	returnValue = Math::lerp(n01, n2, (v2p.magnitude() / v2ab.magnitude())).normalize();
	
	return returnValue;
}

float SimdRenderer::_ggxChi(const float x)
{
	return x > 0 ? 1.0f : 0.0f;
}

float SimdRenderer::_ggxPartial(const Math::Vector4 &v, const Math::Vector4 &h, const Math::Vector4 &n, const float a_2)
{
	const float vDotH = Math::saturate(v.dotProduct(h));
	const float vDotN = Math::saturate(v.dotProduct(n));
	const float chi_f = this->_ggxChi(vDotH / vDotN);
	const float vDotH_2 = vDotH * vDotH;
	const float temporary = (1.0f - vDotH_2) / vDotH_2;
	const float returnValue = (chi_f * 2.0f) / (1.0f + std::sqrt(1.0f + a_2 * temporary));
	
	return returnValue;
}

Math::Vector4 SimdRenderer::_brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v, const float cosinusTheta)
{
	Math::Vector4 returnValue;
	
	// specular color
	const Math::Vector4 c_spec = {1.0f, 1.0f, 1.0f};
	
	// half vector
	const Math::Vector4 h = (l + v).normalized();
	
	// a (alpha) = roughness^2
	const float a = material.roughness * material.roughness;
	const float a_2 = a * a;
	
	// D term (GGX - Trowbridge-Reitz)
	const float nDotH = n.dotProduct(h);
	const float nDotH_2 = nDotH * nDotH;
//	const float chi_d = this->_ggxChi(nDotH);
//	const float denominator = nDotH_2 * a_2 + (1.0f - nDotH_2);
//	const float denominator = (a_2 - 1.0f) * nDotH_2 + 1.0f;
//	const float d = a_2 / (float(M_PI) * denominator * denominator);
	const float acos = std::acos(nDotH);
	const float eTerm = std::pow(float(M_E), (-1.0f * std::pow(std::tan(acos) / a, 2.0f)));
	const float d = (eTerm) / (a_2 * std::pow(a, 4.0f));
	
	// F (fresnel) term (Schlick approximation)
	// FIXME Dialectric materials only
	const Math::Vector4 f01 = {0.0f, 0.0f, 0.0f};
	const Math::Vector4 f0 = Math::lerp(material.color, f01, Math::saturate(material.metallic));
	const Math::Vector4 f = f0 + (Math::Vector4{1.0f, 1.0f, 1.0f} - f0) * std::pow(1.0f - cosinusTheta, 5.0f);
	
	// G term (Schlick-GGX)
	const float g = this->_ggxPartial(v, h, n, a_2) * this->_ggxPartial(l, h, n, a_2);
//	const float vDotH = Math::saturate(v.dotProduct(h));
//	const float vDotN = Math::saturate(v.dotProduct(n));
//	const float temp = (2.0f * n.dotProduct(h)) / v.dotProduct(h);
//	const float g0 = 1.0f;
//	const float g1 = temp * n.dotProduct(v);
//	const float g2 = temp * n.dotProduct(l);
//	const float g = std::min(g0, std::min(g1, g2));
	
	returnValue = Math::saturate((d * f * g) / (4 * n.dotProduct(l) * n.dotProduct(v)));
	
//	if (returnValue != Math::Vector4{0.0f, 0.0f, 0.0f})
//	{
//		std::stringstream stream;
//		stream << "a^2 " << a_2 << "\n";
//		stream << "d " << d << "\n";
////		stream << "d1 " << d1 << "\n";
////		stream << "g " << g << "\n";
////		stream << "f " << f << "\n";
//		stream << returnValue << "\n\n";
//		std::cout << stream.str();
//	}
	
	return returnValue;
}

}
