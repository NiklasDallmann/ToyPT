#include <algorithm>
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

#include "renderer.h"

namespace Rendering
{

Renderer::Renderer()
{
}

void Renderer::render(FrameBuffer &frameBuffer, const float fieldOfView, const size_t samples, const size_t bounces)
{
	const size_t width = frameBuffer.width();
	const size_t height = frameBuffer.height();
	float fovRadians = fieldOfView / 180.0f * float(M_PI);
	float zCoordinate = -(width/(2.0f * std::tan(fovRadians / 2.0f)));
	size_t linesFinished = 0;
	std::stringstream stream;
	std::chrono::time_point<std::chrono::high_resolution_clock> begin = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end = begin;

#pragma omp parallel for schedule(dynamic)
	for (size_t j = 0; j < height; j++)
	{
		for (size_t i = 0; i < width; i++)
		{
			float x = (i + 0.5f) - (width / 2.0f);
			float y = -(j + 0.5f) + (height / 2.0f);
			
			Math::Vector4 direction{x, y, zCoordinate};
			direction.normalize();
			
			Math::Vector4 color;
			
			for (size_t sample = 0; sample < samples; sample++)
			{
				color += this->_castRay(direction, {0, 0, 0}, bounces);
			}
			
			frameBuffer.pixel(i, j) = (color / float(samples));
		}
		
#pragma omp critical
		{
			std::chrono::time_point<std::chrono::high_resolution_clock> current = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> elapsed = current - begin;
			linesFinished++;
			stream << std::setw(4) << std::setfill('0') << std::fixed << std::setprecision(1) << (float(linesFinished) / float(height) * 100.0f) << "% " << elapsed.count() << "s\r";
			std::cout << stream.str() << std::flush;
		}
	}
	
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> elapsed = end - begin;
	stream << std::setw(4) << std::setfill('0') << std::fixed << std::setprecision(1) << (float(linesFinished) / float(height) * 100.0f) << "% " << elapsed.count() << "s\n";
	std::cout << stream.str();
}

bool Renderer::_intersectTriangle(const float distance, const Math::Vector4 &direction, const Math::Vector4 &origin, const Triangle *triangle)
{
	bool returnValue = false;
	
	Vertex v0, v1, v2;
	v0 = this->vertexBuffer[triangle->vertices[0]];
	v1 = this->vertexBuffer[triangle->vertices[1]];
	v2 = this->vertexBuffer[triangle->vertices[2]];
	
	Math::Vector4 n = Triangle::normal(triangle, this->vertexBuffer.data());
	Math::Vector4 p = origin + distance * direction;
	Math::Vector4 e01 = v1 - v0;
	Math::Vector4 e12 = v2 - v1;
	Math::Vector4 e20 = v0 - v2;
	Math::Vector4 e0p = p - v0;
	Math::Vector4 e1p = p - v1;
	Math::Vector4 e2p = p - v2;
	
	float conditionInsideE01 = e01.crossProduct(e0p).dotProduct(n);
	float conditionInsideE12 = e12.crossProduct(e1p).dotProduct(n);
	float conditionInsideE20 = e20.crossProduct(e2p).dotProduct(n);
	
	returnValue = (conditionInsideE01 >= 0.0f & conditionInsideE12 >= 0.0f & conditionInsideE20 >= 0.0f);
	
	return returnValue;
}

float Renderer::_intersectPlane(const Math::Vector4 &direction, const Math::Vector4 &origin, const Triangle *triangle)
{
	float returnValue = 0;
	
	Math::Vector4 n = Triangle::normal(triangle, this->vertexBuffer.data());
	float t0 = (this->vertexBuffer[triangle->vertices[0]] - origin).dotProduct(n);
	float t1 = n.dotProduct(direction);
	
	if (t1 <= _epsilon & t1 >= -_epsilon)
	{
		returnValue = -1.0f;
		goto exit;
	}
	
	returnValue = t0 / t1;
	
exit:
	return returnValue;
}

__m256 Renderer::_intersectPlaneSimd(const Math::Vector4 &direction, const Math::Vector4 &origin, const Triangle *triangles, const __m256 normals)
{
	__m256 returnValue = {0.0f, 0.0f, 0.0f, 0.0f};
	
//	__m256 t0 = 
	
	return returnValue;
}

float Renderer::_traceRay(const Math::Vector4 &direction, const Math::Vector4 &origin, IntersectionInfo &intersection)
{
	float returnValue = 0.0f;
	
	Mesh *nearestMesh = nullptr;
	Triangle *nearestTriangle = nullptr;
	float planeDistance = 0.0f;
	float distance = std::numeric_limits<float>::max();
	planeDistance = distance;
	
	for (Mesh &mesh : this->meshBuffer)
	{
		for (uint32_t triangleIndex = mesh.triangleOffset; triangleIndex < (mesh.triangleOffset + mesh.triangleCount); triangleIndex++)
		{
			Triangle *triangle = &this->triangleBuffer[triangleIndex];
			planeDistance = this->_intersectPlane(direction, origin, triangle);
			
			if ((planeDistance > _epsilon) & (planeDistance < distance) & this->_intersectTriangle(planeDistance, direction, origin, triangle))
			{
				distance = planeDistance;
				nearestMesh = &mesh;
				nearestTriangle = triangle;
			}
		}
	}
	
	returnValue = distance;
	intersection.mesh = nearestMesh;
	intersection.triangle = nearestTriangle;
	
	return returnValue;
}

Math::Vector4 Renderer::_castRay(const Math::Vector4 &direction, const Math::Vector4 &origin, const size_t maxBounces)
{
	Math::Vector4 returnValue = {0.0f, 0.0f, 0.0f};
	Math::Vector4 mask = {1.0f, 1.0f, 1.0f};
	
	Math::Vector4 currentDirection = direction;
	Math::Vector4 currentOrigin = origin;
	
	// FIXME This won't stay constant
	float pdf = (2.0f * float(M_PI));
	float r1 = 1.0f;
	float r2 = 1.0f;
	
	std::random_device device;
	std::default_random_engine generator(device());
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	Math::Vector4 Nt;
	Math::Vector4 Nb;
	
	for (size_t currentBounce = 0; currentBounce < maxBounces; currentBounce++)
	{
		Math::Vector4 intersectionPoint;
		IntersectionInfo intersection;
		Math::Vector4 normal;
		float distance = this->_traceRay(currentDirection, currentOrigin, intersection);
		intersectionPoint = currentOrigin + distance * currentDirection;
		
		if (intersection.triangle != nullptr)
		{
			Math::Vector4 color = this->materialBuffer[intersection.mesh->materialOffset].color();
			Math::Vector4 directLight = {0.0f, 0.0f, 0.0f};
			normal = Triangle::normal(intersection.triangle, this->vertexBuffer.data());
			
			// Intersection found
			for (const PointLight &pointLight : this->pointLightBuffer)
			{
				IntersectionInfo occlusionIntersection;
				const Math::Vector4 lightDirection = pointLight.position() - intersectionPoint;
				const float occlusionDistance = this->_traceRay(lightDirection.normalized(), intersectionPoint, occlusionIntersection);
				const bool visible = ((occlusionIntersection.triangle == nullptr | occlusionDistance > lightDirection.magnitude()) &
									  ((normal.dotProduct(lightDirection)) > 0.0f));
				
				directLight += ((normal.dotProduct(lightDirection.normalized())) * pointLight.color()) * visible;
			}
			
			directLight = directLight * color;
//			directLight = color;
			
			// Indirect lighting
			this->_createCoordinateSystem(normal, Nt, Nb);
			
			// Generate hemisphere
			r1 = distribution(generator);
			r2 = distribution(generator);
			float sinTheta = std::pow((1.0f - r1 * r1), 0.5f);
			float phi = 2.0f * float(M_PI) * r2;
			float x = sinTheta * std::cos(phi);
			float z = sinTheta * std::sin(phi);
			Math::Vector4 sampleHemisphere{x, r1, z};
			
			Math::Matrix4x4 matrix{
				{Nb.x(), normal.x(), Nt.x()},
				{Nb.y(), normal.y(), Nt.y()},
				{Nb.z(), normal.z(), Nt.z()}
			};
			
			Math::Vector4 sampleWorld = matrix * sampleHemisphere;
			
			currentDirection = (intersectionPoint + sampleWorld).normalized();
			currentOrigin = sampleWorld;
			
			returnValue += mask * directLight;
			mask = mask * color;
			mask = mask * r1 * pdf;
		}
		else
		{
			break;
		}
	}
	
	return returnValue;
}

void Renderer::_createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal)
{
	const Math::Vector4 a = Math::Vector4{normal.z(), 0.0f, -normal.x()};
	const Math::Vector4 b = Math::Vector4{0.0f, -normal.z(), normal.y()};
	float t = std::abs(normal.x()) > std::fabs(normal.y());
	
	tangentNormal = Math::lerp(a, b, t).normalized();
	
	binormal = normal.crossProduct(tangentNormal);
}

float Renderer::_brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v)
{
	float returnValue = 1.0f;
	
//	const Math::Vector4 h = (l + v).normalized();
//	const float a_2 = std::pow(material.roughness(), 4.0f);
//	const float d = a_2 / (float(M_PI) * std::pow((std::pow(n * h, 2.0f) * (a_2 - 1) + 1), 2.0f));
//	const float f = 1.0f;
//	const float g = 1.0f;
	
//	returnValue = (d * f * g) / (4.0f * (n * l) * (n * v));
	
	return returnValue;
}

}
