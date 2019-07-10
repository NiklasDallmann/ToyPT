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

void Renderer::setMeshes(const std::vector<Mesh> &meshes)
{
	this->_meshes = meshes;
}

void Renderer::setPointLights(const std::vector<PointLight> &pointLights)
{
	this->_pointLights = pointLights;
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

bool Renderer::_intersectTriangle(const float distance, const Math::Vector4 &direction, const Math::Vector4 &origin, const Triangle &triangle, const Math::Vector4 &normal)
{
	bool returnValue = false;
//	Math::Vector3D n = triangle.normal();
	Math::Vector4 p = origin + distance * direction;
	Math::Vector4 e01 = triangle[1] - triangle[0];
	Math::Vector4 e12 = triangle[2] - triangle[1];
	Math::Vector4 e20 = triangle[0] - triangle[2];
	Math::Vector4 e0p = p - triangle[0];
	Math::Vector4 e1p = p - triangle[1];
	Math::Vector4 e2p = p - triangle[2];
	
	float conditionInsideE01 = e01.crossProduct(e0p) * normal;
	float conditionInsideE12 = e12.crossProduct(e1p) * normal;
	float conditionInsideE20 = e20.crossProduct(e2p) * normal;
	
	if (conditionInsideE01 >= 0.0f & conditionInsideE12 >= 0.0f & conditionInsideE20 >= 0.0f)
	{
		// Intersection found
		returnValue = true;
	}
	else
	{
		returnValue = false;
	}
	
	return returnValue;
}

float Renderer::_intersectPlane(const Math::Vector4 &direction, const Math::Vector4 &origin, const Triangle &triangle, const Math::Vector4 &normal)
{
	float returnValue = 0;
	
//	Math::Vector3D n = triangle.normal();
	float t0 = (triangle[0] - origin) * normal;
	float t1 = normal * direction;
	
	if (t1 <= _epsilon & t1 >= -_epsilon)
	{
		returnValue = -1.0f;
		goto exit;
	}
	
	returnValue = t0 / t1;
	
exit:
	return returnValue;
}

__m256 Renderer::_intersectPlaneSimd(const Math::Vector4 &direction, const Math::Vector4 &origin, const Triangle *triangles, const Math::Vector4 &normal)
{
	__m256 returnValue = {0.0f, 0.0f, 0.0f, 0.0f};
	
	
	
	return returnValue;
}

float Renderer::_traceRay(const Math::Vector4 &direction, const Math::Vector4 &origin, IntersectionInfo &intersection)
{
	float returnValue = 0.0f;
	
	Mesh *nearestMesh = nullptr;
	Triangle *nearestTriangle = nullptr;
	Math::Vector4 normal = 0.0f;
	float planeDistance = 0.0f;
	float distance = std::numeric_limits<float>::max();
	planeDistance = distance;
	
	for (Mesh &mesh : this->_meshes)
	{
		for (Triangle &triangle : mesh.triangles())
		{
			normal = triangle.normal();
			planeDistance = this->_intersectPlane(direction, origin, triangle, normal);
			
			if ((planeDistance > _epsilon) & (planeDistance < distance) & this->_intersectTriangle(planeDistance, direction, origin, triangle, normal))
			{
				distance = planeDistance;
				nearestMesh = &mesh;
				nearestTriangle = &triangle;
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
			Math::Vector4 color = intersection.mesh->material().color();
			Math::Vector4 directLight = {0.0f, 0.0f, 0.0f};
			normal = intersection.triangle->normal();
			
			// Intersection found
			for (const PointLight &pointLight : this->_pointLights)
			{
				IntersectionInfo occlusionIntersection;
				const Math::Vector4 lightDirection = pointLight.position() - intersectionPoint;
				const float occlusionDistance = this->_traceRay(lightDirection.normalized(), intersectionPoint, occlusionIntersection);
				const bool visible = ((occlusionIntersection.triangle == nullptr | occlusionDistance > lightDirection.magnitude()) &
									  ((normal * lightDirection) > 0.0f));
				
				directLight += ((normal * lightDirection.normalized()) * pointLight.color()) * visible;
			}
			
			directLight = directLight.coordinateProduct(color);
			
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
			
			returnValue += mask.coordinateProduct(directLight);
			mask = mask.coordinateProduct(color);
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
	if (std::abs(normal.x()) > std::fabs(normal.y()))
	{
		tangentNormal = Math::Vector4{normal.z(), 0.0f, -normal.x()}.normalized();
	}
	else
	{
		tangentNormal = Math::Vector4{0.0f, -normal.z(), normal.y()}.normalized();
	}
	
	binormal = normal.crossProduct(tangentNormal);
}

float Renderer::_brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v)
{
	float returnValue = 1.0f;
	
	const Math::Vector4 h = (l + v).normalized();
	const float a_2 = std::pow(material.roughness(), 4.0f);
	const float d = a_2 / (float(M_PI) * std::pow((std::pow(n * h, 2.0f) * (a_2 - 1) + 1), 2.0f));
	const float f = 1.0f;
	const float g = 1.0f;
	
	returnValue = (d * f * g) / (4.0f * (n * l) * (n * v));
	
	return returnValue;
}

}
