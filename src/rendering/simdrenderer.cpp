#include <algorithm>
#include <simdvector3pack.h>
#include <chrono>
#include <cmath>
#include <map>
#include <matrix4x4.h>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <iomanip>
#include <iostream>
#include <sstream>

#include "simdrenderer.h"

namespace Rendering
{

SimdRenderer::SimdRenderer()
{
}

void SimdRenderer::render(FrameBuffer &frameBuffer, Obj::GeometryContainer &geometry, const CallBack &callBack, const bool &abort, const float fieldOfView, const uint32_t samples, const uint32_t bounces, const Math::Vector4 &skyColor)
{
	const uint32_t width = frameBuffer.width();
	const uint32_t height = frameBuffer.height();
	float fovRadians = fieldOfView / 180.0f * float(M_PI);
	float zCoordinate = -(width/(2.0f * std::tan(fovRadians / 2.0f)));
	std::stringstream stream;
	std::chrono::time_point<std::chrono::high_resolution_clock> begin = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end = begin;
	std::chrono::duration<float> elapsed;
	
	this->_geometryToBuffer(geometry, this->_triangleBuffer, this->_meshBuffer);
	
	std::random_device device;
	
	for (size_t sample = 1; (sample <= samples) & ~abort; sample++)
	{
#pragma omp parallel for schedule(dynamic, 20) collapse(2)
		for (uint32_t h = 0; h < height; h++)
		{
			for (uint32_t w = 0; w < width; w++)
			{
				RandomNumberGenerator rng(device());
				float offsetX, offsetY;
				const float scalingFactor = 1.0f / float(std::numeric_limits<uint32_t>::max());
				offsetX = rng.get(scalingFactor)  - 0.5f;
				offsetY = rng.get(scalingFactor) - 0.5f;
				
				float x = (w + offsetX + 0.5f) - (width / 2.0f);
				float y = -(h + offsetY + 0.5f) + (height / 2.0f);
				
				Math::Vector4 direction{x, y, zCoordinate};
				direction.normalize();
				
				Math::Vector4 color = frameBuffer.pixel(w, h) * float(sample - 1);
				
				color += this->_castRay({{0, 0, 0}, direction}, geometry, rng, bounces, skyColor);
				
				frameBuffer.setPixel(w, h, (color / float(sample)));
			}
		}
		
		callBack();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		stream << std::setw(3) << std::setfill('0') << sample << "/" << samples << " samples; " << elapsed.count() << " seconds\r";
		std::cout << stream.str() << std::flush;
	}
	
	std::cout << std::endl;
}

void SimdRenderer::_geometryToBuffer(const Obj::GeometryContainer &geometry, Simd::PreComputedTriangleBuffer &triangleBuffer, Simd::MeshBuffer &meshBuffer)
{
	for (uint32_t meshIndex = 0; meshIndex < geometry.meshBuffer.size(); meshIndex++)
	{
		const Obj::Mesh &objMesh = geometry.meshBuffer[meshIndex];
		Simd::Mesh mesh;
		mesh.triangleOffset = triangleBuffer.size();
		mesh.triangleCount = objMesh.triangleCount;
		mesh.materialOffset = objMesh.materialOffset;
		
		for (uint32_t triangleIndex = 0; triangleIndex < objMesh.triangleCount; triangleIndex++)
		{
			const Obj::Triangle &triangle = geometry.triangleBuffer[objMesh.triangleOffset + triangleIndex];
			
			Simd::Vertex v0, v1, v2;
			Simd::Normal n0, n1, n2;
			
			v0 = geometry.vertexBuffer[triangle.vertices[0]];
			v1 = geometry.vertexBuffer[triangle.vertices[1]];
			v2 = geometry.vertexBuffer[triangle.vertices[2]];
			
			n0 = geometry.normalBuffer[triangle.normals[0]];
			n1 = geometry.normalBuffer[triangle.normals[1]];
			n2 = geometry.normalBuffer[triangle.normals[2]];
			
			triangleBuffer.append(v0, v1, v2, n0, n1, n2, Simd::maskTrue);
		}
		
		meshBuffer.push_back(mesh);
	}
	
	const uint32_t triangleBufferSize = uint32_t(triangleBuffer.size());
	const uint32_t paddingTriangles = Simd::avx2FloatCount - (triangleBufferSize % Simd::avx2FloatCount);
	
	for (uint32_t i = 0; i < paddingTriangles; i++)
	{
		triangleBuffer.append({}, {}, {}, {}, {}, {}, Simd::maskFalse);
	}
}

__m256 SimdRenderer::_intersectSimd(const Ray &ray, Simd::PrecomputedTrianglePointer &data, __m256 &ts, __m256 &us, __m256 &vs)
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
	
	epsilon = _mm256_set1_ps(_epsilon);
	
	// Load vertex coordinates
	v0.loadUnaligned(data.v0.x, data.v0.y, data.v0.z);
	v1.loadUnaligned(data.v1.x, data.v1.y, data.v1.z);
	v2.loadUnaligned(data.v2.x, data.v2.y, data.v2.z);
	e01.loadUnaligned(data.e01.x, data.e01.y, data.e01.z);
	e02.loadUnaligned(data.e02.x, data.e02.y, data.e02.z);
	
	data += Simd::avx2FloatCount;
	
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

template <SimdRenderer::TraceType T>
float SimdRenderer::_traceRay(const Ray &ray, const Obj::GeometryContainer &geometry, IntersectionInfo &intersection)
{
	float returnValue = 0.0f;
	
	uint32_t triangleCount = this->_triangleBuffer.size();
	bool intersectionFound = false;
	Simd::Mesh *nearestMesh = nullptr;
	uint32_t nearestTriangle = 0xFFFFFFFF;
	float newDistance = 0.0f;
	float distance = std::numeric_limits<float>::max();
	float u = 0;
	float v = 0;
	newDistance = distance;
	
	// Intersect triangles
	Simd::PrecomputedTrianglePointer dataPointer = this->_triangleBuffer.data();
	
	uint32_t triangleIndex = 0;
	uint32_t avx2Loops = triangleCount - (triangleCount % Simd::avx2FloatCount);
	
	for (; triangleIndex < avx2Loops; triangleIndex += Simd::avx2FloatCount)
	{
		__m256 ts, us, vs;
		__m256 intersected = this->_intersectSimd(ray, dataPointer, ts, us, vs);
		
		for (uint32_t index = 0; index < Simd::avx2FloatCount; index++)
		{
			newDistance = ts[index];
			if ((newDistance < distance) & bool(intersected[index]))
			{
				intersectionFound = true;
				distance = newDistance;
				nearestTriangle = triangleIndex + index;
				
				if constexpr (T == TraceType::Light)
				{
					for (uint32_t meshIndex = 0; meshIndex < this->_meshBuffer.size(); meshIndex++)
					{
						Simd::Mesh &mesh = this->_meshBuffer[meshIndex];
						
						if ((nearestTriangle >= mesh.triangleOffset) &
							(nearestTriangle < (mesh.triangleOffset + mesh.triangleCount)) &
							(geometry.materialBuffer[mesh.materialOffset].emittance() > 0.0f))
						{
							nearestMesh = &mesh;
							break;
						}
					}
				}
			}
		}
	}
	
	// Find corresponding mesh
	if constexpr (T == TraceType::Object)
	{
		if (intersectionFound)
		{
			for (uint32_t meshIndex = 0; meshIndex < this->_meshBuffer.size(); meshIndex++)
			{
				Simd::Mesh &mesh = this->_meshBuffer[meshIndex];
				
				if ((nearestTriangle >= mesh.triangleOffset) &
					(nearestTriangle < (mesh.triangleOffset + mesh.triangleCount)))
				{
					nearestMesh = &mesh;
					break;
				}
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

Math::Vector4 SimdRenderer::_castRay(const Ray &ray, const Obj::GeometryContainer &geometry, RandomNumberGenerator rng, const size_t maxBounces, const Math::Vector4 &skyColor)
{
	Math::Vector4 returnValue = {0.0f, 0.0f, 0.0f};
	
	// Multiply colors
	Math::Vector4 mask = {1.0f, 1.0f, 1.0f};
	
	Math::Vector4 currentDirection = ray.direction;
	Math::Vector4 currentOrigin = ray.origin;
	
	// FIXME This won't stay constant
//	float pdf = 2.0f * float(M_PI);
	float cosinusTheta;
	
	for (size_t currentBounce = 0; currentBounce < maxBounces; currentBounce++)
	{
		Math::Vector4 intersectionPoint;
		IntersectionInfo objectIntersection;
		Math::Vector4 normal;
		float distance = this->_traceRay<TraceType::Object>({currentOrigin, currentDirection}, geometry, objectIntersection);
		
		intersectionPoint = currentOrigin + (distance * currentDirection);
		
		if (objectIntersection.mesh != nullptr)
		{
			const Material &objectMaterial = geometry.materialBuffer[objectIntersection.mesh->materialOffset];
			const Math::Vector4 &objectColor = objectMaterial.color();
			
			// Calculate normal
			Simd::PrecomputedTrianglePointer dataPointer = this->_triangleBuffer.data() + objectIntersection.triangleOffset;
			normal = this->_interpolateNormal(intersectionPoint, dataPointer);
			
			// Calculate new origin and offset
			currentOrigin = intersectionPoint + (_epsilon * normal);
			
			// Direct illumination
			Math::Vector4 newDirection, reflectedDirection, diffuseDirection, directIllumination;
			IntersectionInfo lightIntersection;
			
			if (objectMaterial.roughness() > 0.0f)
			{
				diffuseDirection = this->_randomDirection(normal, rng, cosinusTheta);
				this->_traceRay<TraceType::Light>({currentOrigin, diffuseDirection}, geometry, lightIntersection);
				
				if (lightIntersection.mesh != nullptr)
				{
					const Material &lightMaterial = geometry.materialBuffer[lightIntersection.mesh->materialOffset];
					const Math::Vector4 &lightColor = lightMaterial.color();
					
					directIllumination = objectColor * lightColor * lightMaterial.emittance();
				}
			}
			
			// Global illumination
			Math::Vector4 diffuse, specular;
			
			diffuseDirection = this->_randomDirection(normal, rng, cosinusTheta);
			reflectedDirection = (currentDirection - 2.0f * currentDirection.dotProduct(normal) * normal).normalize();
			
			newDirection = Math::lerp(diffuseDirection, reflectedDirection, objectMaterial.roughness());
			
			diffuse = 2.0f * objectColor;
//			specular = this->_brdf(material, normal, newDirection, currentDirection);
			
			currentDirection = newDirection;
			
//			returnValue += objectColor * objectMaterial.emittance() * mask;
			returnValue += directIllumination;
			mask *= (diffuse) * cosinusTheta;
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
	cosinusTheta = rng.get(scalingFactor);
	ratio = rng.get(scalingFactor);
	
	Math::Vector4 sample = this->_createUniformHemisphere(cosinusTheta, ratio);
	
	Math::Matrix4x4 localToWorldMatrix{
		{Nb.x(), normal.x(), Nt.x()},
		{Nb.y(), normal.y(), Nt.y()},
		{Nb.z(), normal.z(), Nt.z()}
	};
	
	return (localToWorldMatrix * sample).normalize();
}

Math::Vector4 SimdRenderer::_interpolateNormal(const Math::Vector4 &intersectionPoint, Simd::PrecomputedTrianglePointer &data)
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
	
	denominator = (e01.x() * v2p.y() - v2p.x() * e01.y()) + _epsilon;
	a = (-(v0.x() * v2p.y() - v2p.x() * v0.y() + v2p.x() * v2.y() - v2.x() * v2p.y())) / denominator;
	b = (e01.x() * v0.y() - e01.x() * v2.y() - v0.x() * e01.y() + v2.x() * e01.y()) / denominator;
	
	vab = v0 + a * e01;
	
	n01 = Math::lerp(n1, n0, a).normalize();
	v2ab = vab - v2;
	
	returnValue = Math::lerp(n01, n2, (v2p.magnitude() / v2ab.magnitude())).normalize();
	
	return returnValue;
}

Math::Vector4 SimdRenderer::_brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v)
{
	Math::Vector4 returnValue;
	
	// specular color
	const Math::Vector4 c_spec = {1.0f, 1.0f, 1.0f};
	
	// half vector
	const Math::Vector4 h = (l + v).normalized();
	
	// a (alpha) = roughness^2
	const float a = std::pow(material.roughness(), 2.0f);
	const float a_2 = std::pow(material.roughness(), 4.0f);
	
	// D term (GGX - Trowbridge-Reitz)
	const float d = a_2 /
			(float(M_PI) * std::pow((std::pow(n.dotProduct(h), 2.0f) * (a_2 - 1.0f) + 1.0f), 2.0f));
	
	// F (fresnel) term (Schlick approximation)
	const Math::Vector4 f = c_spec + ((1.0f - c_spec) * std::pow((1.0f - l.dotProduct(h)), 5.0f));
	
	// G term (Schlick-GGX)
	const float k = a / 2.0f;
	const float g = n.dotProduct(v) / (n.dotProduct(v) * (1.0f - k) + k);
	
	returnValue = (d * f * g) / (4.0f * (n.dotProduct(l)) * (n.dotProduct(v)));
	
	// [0, 1] |-> [1, MAX]
//	float alpha = (512.0f * std::pow(1.953125E-3f, material.roughness()));
//	float alpha = 15.0f;
//	float alpha = material.roughness();
//	Math::Vector4 reflected = v - 2.0f * v.dotProduct(n) * n;
	
//	returnValue = c_spec * (alpha + 2.0f) * std::pow(v.dotProduct(reflected), alpha);
	
	return returnValue;
}

}
