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
//				bool debug = (i == width / 2) & (j == 3 * (height / 4)) & (sample == 0);
				color += this->_castRay({{0, 0, 0}, direction}, bounces);
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

bool Renderer::_intersectMoellerTrumbore(const Ray &ray, const Triangle *triangle, float &t, float &u, float &v)
{
	bool returnValue = true;
	
	const Math::Vector4 v0 = this->vertexBuffer[triangle->vertices[0]];
	const Math::Vector4 v1 = this->vertexBuffer[triangle->vertices[1]];
	const Math::Vector4 v2 = this->vertexBuffer[triangle->vertices[2]];
	
	const Math::Vector4 v01 = v1 - v0;
	const Math::Vector4 v02 = v2 - v0;
	Math::Vector4 pVector = ray.direction.crossProduct(v02);
	Math::Vector4 v0o;
	Math::Vector4 qVector;
	const float determinant = v01.dotProduct(pVector);
	const float inverseDeterminant = 1.0f / determinant;
	
	v0o = ray.origin - v0;
	u = v0o.dotProduct(pVector) * inverseDeterminant;
	
	qVector = v0o.crossProduct(v01);
	v = ray.direction.dotProduct(qVector) * inverseDeterminant;
	
	t = v02.dotProduct(qVector) * inverseDeterminant;
	
	returnValue = ~((determinant < _epsilon) | (u < 0.0f) | (u > 1.0f) | (v < 0.0f) | ((u + v) > 1.0f)) & (t > _epsilon);
	
	return returnValue;
}

float Renderer::_traceRay(const Ray &ray, IntersectionInfo &intersection)
{
	float returnValue = 0.0f;
	
	Mesh *nearestMesh = nullptr;
	Triangle *nearestTriangle = nullptr;
	float newDistance = 0.0f;
	float distance = std::numeric_limits<float>::max();
	float u = 0;
	float v = 0;
	newDistance = distance;
	
	for (Mesh &mesh : this->meshBuffer)
	{
		for (uint32_t triangleIndex = mesh.triangleOffset; triangleIndex < (mesh.triangleOffset + mesh.triangleCount); triangleIndex++)
		{
			Triangle *triangle = &this->triangleBuffer[triangleIndex];
			
			bool intersected = this->_intersectMoellerTrumbore(ray, triangle, newDistance, u, v);
			
			if ((newDistance < distance) & intersected)
			{
				distance = newDistance;
				nearestMesh = &mesh;
				nearestTriangle = triangle;
			}
		}
	}
	
	returnValue = distance;
	intersection.mesh = nearestMesh;
	intersection.triangle = nearestTriangle;
	intersection.u = u;
	intersection.v = v;
	
	return returnValue;
}

Math::Vector4 Renderer::_castRay(const Ray &ray, const size_t maxBounces, const bool debug)
{
	Math::Vector4 returnValue = {0.0f, 0.0f, 0.0f};
	
	// Multiply colors
	Math::Vector4 mask = {1.0f, 1.0f, 1.0f};
	
	// Emitted light: += (mask * (color * emittance))
	Math::Vector4 emittedLight = {0.0f, 0.0f, 0.0f};
	
	Math::Vector4 currentDirection = ray.direction;
	Math::Vector4 currentOrigin = ray.origin;
	
	// FIXME This won't stay constant
	float pdf = 2.0f * float(M_PI);
	float cosinusTheta = 1.0f;
	float ratio = 1.0f;
	
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
		float distance = this->_traceRay({currentOrigin, currentDirection}, intersection);
		
		intersectionPoint = currentOrigin + (distance * currentDirection);
		
		if (intersection.triangle != nullptr)
		{
			Material &material = this->materialBuffer[intersection.mesh->materialOffset];
			Math::Vector4 color = material.color();
			
			// Face normal
//			normal = Triangle::normal(intersection.triangle, this->vertexBuffer.data());
			
			normal = this->_interpolateNormal(intersection, intersectionPoint);
			
			// Indirect lighting
			this->_createCoordinateSystem(normal, Nt, Nb);
			
			// Generate hemisphere
			cosinusTheta = distribution(generator);
			ratio = distribution(generator);
			Math::Vector4 sample = this->_createUniformHemisphere(cosinusTheta, ratio);
			
			Math::Matrix4x4 localToWorldMatrix{
				{Nb.x(), normal.x(), Nt.x()},
				{Nb.y(), normal.y(), Nt.y()},
				{Nb.z(), normal.z(), Nt.z()}
			};
			
			Math::Vector4 newDirection = (localToWorldMatrix * sample).normalized();
			currentOrigin = intersectionPoint + (_epsilon * normal);
			
//			Math::Vector4 brdf = this->_brdf(material, normal, newDirection, currentDirection);
			currentDirection = newDirection;
			
			Math::Vector4 diffuse, specular;
			diffuse = 2.0f * color;
//			specular = brdf;
			
			emittedLight += color * material.emittance() * mask;
			mask *= (diffuse) * cosinusTheta;
		}
		else
		{
			break;
		}
	}
	
	returnValue = emittedLight;
	
	return returnValue;
}

void Renderer::_createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal)
{
	const Math::Vector4 a = Math::Vector4{normal.z(), 0.0f, -normal.x()};
	const Math::Vector4 b = Math::Vector4{0.0f, -normal.z(), normal.y()};
	float t = std::abs(normal.x()) > std::abs(normal.y());
	
	tangentNormal = Math::lerp(a, b, t).normalized();
	
	binormal = normal.crossProduct(tangentNormal);
}

Math::Vector4 Renderer::_createUniformHemisphere(const float r1, const float r2)
{
	float sinTheta = std::sqrt(1.0f - r1 * r1);
	float phi = 2.0f * float(M_PI) * r2;
	float x = sinTheta * std::cos(phi);
	float z = sinTheta * std::sin(phi);
	return {x, r1, z};
}

Math::Vector4 Renderer::_interpolateNormal(const IntersectionInfo &intersection, const Math::Vector4 &intersectionPoint)
{
	Math::Vector4 returnValue, p, n0, n1, n2, n01, n02, v0, v1, v2, v01, v02, v12, v0p, v1p, v2p, vab, v2ab;
	
	n0 = this->normalBuffer[intersection.triangle->normals[0]];
	n1 = this->normalBuffer[intersection.triangle->normals[1]];
	n2 = this->normalBuffer[intersection.triangle->normals[2]];
	
	v0 = this->vertexBuffer[intersection.triangle->vertices[0]];
	v1 = this->vertexBuffer[intersection.triangle->vertices[1]];
	v2 = this->vertexBuffer[intersection.triangle->vertices[2]];
	
	p = intersectionPoint;
	v01 = v1 - v0;
	v02 = v2 - v0;
	v12 = v2 - v1;
	v0p = p - v0;
	v1p = p - v1;
	v2p = p - v2;
	
	float a, b, denominator;
	
	denominator = (v01.x() * v2p.y() - v2p.x() * v01.y()) + _epsilon;
	a = (-(v0.x() * v2p.y() - v2p.x() * v0.y() + v2p.x() * v2.y() - v2.x() * v2p.y())) / denominator;
	b = (v01.x() * v0.y() - v01.x() * v2.y() - v0.x() * v01.y() + v2.x() * v01.y()) / denominator;
	
	vab = v0 + a * v01;
	
	n01 = Math::lerp(n1, n0, a).normalize();
	v2ab = vab - v2;
	
	returnValue = Math::lerp(n01, n2, (v2p.magnitude() / v2ab.magnitude())).normalize();
	
	return returnValue;
}

Math::Vector4 Renderer::_brdf(const Material &material, const Math::Vector4 &n, const Math::Vector4 &l, const Math::Vector4 &v)
{
	Math::Vector4 returnValue;
	
	// specular color
	const Math::Vector4 c_spec = {1.0f, 1.0f, 1.0f};
	
//	// half vector
//	const Math::Vector4 h = (l + v).normalized();
	
//	// a (alpha) = roughness^2
//	const float a = std::pow(material.roughness(), 2.0f);
//	const float a_2 = std::pow(material.roughness(), 4.0f);
	
//	// D term (GGX - Trowbridge-Reitz)
//	const float d = a_2 /
//			(float(M_PI) * std::pow((std::pow(n.dotProduct(h), 2.0f) * (a_2 - 1.0f) + 1.0f), 2.0f));
	
//	// F (fresnel) term (Schlick approximation)
//	const Math::Vector4 f = c_spec + ((1.0f - c_spec) * std::pow((1.0f - l.dotProduct(h)), 5.0f));
	
//	// G term (Schlick-GGX)
//	const float k = a / 2.0f;
//	const float g = n.dotProduct(v) / (n.dotProduct(v) * (1.0f - k) + k);
	
//	returnValue = (d * f * g) / (4.0f * (n.dotProduct(l)) * (n.dotProduct(v)));
	
	// [0, 1] |-> [1, MAX]
//	float alpha = (512.0f * std::pow(1.953125E-3f, material.roughness()));
	float alpha = 300.0f;
	
	returnValue = (alpha + 2.0f) * c_spec * std::pow(v.dotProduct(l), alpha);
	
	return returnValue;
}

}
