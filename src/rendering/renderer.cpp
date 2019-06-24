#include <algorithm>
#include <cmath>
#include <map>
#include <matrix3d.h>
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

void Renderer::setMeshes(const std::vector<AbstractMesh *> &meshes)
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

#pragma omp parallel for schedule(dynamic)
	for (size_t j = 0; j < height; j++)
	{
		for (size_t i = 0; i < width; i++)
		{
			float x = (i + 0.5f) - (width / 2.0f);
			float y = -(j + 0.5f) + (height / 2.0f);
			
			Math::Vector3D direction{x, y, zCoordinate};
			direction.normalize();
			
			Math::Vector3D color;
			
//#pragma omp parallel for
			for (size_t sample = 0; sample < samples; sample++)
			{
				color += this->_castRay(direction, {0, 0, 0}, 0, bounces);
			}
			
			frameBuffer.pixel(i, j) = (color / float(samples));
		}
		
#pragma omp critical
		{
			linesFinished++;
			stream << std::setw(4) << std::setfill('0') << std::fixed << std::setprecision(1) << (float(linesFinished) / float(height) * 100.0f) << "%\r";
			std::cout << stream.str() << std::flush;
		}
	}
	
	stream << std::setw(4) << std::setfill('0') << std::fixed << std::setprecision(1) << (float(linesFinished) / float(height) * 100.0f) << "%\n";
	std::cout << stream.str();
}

bool Renderer::_intersectTriangle(const float distance, const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle, const Math::Vector3D &normal)
{
	bool returnValue = false;
//	Math::Vector3D n = triangle.normal();
	Math::Vector3D p = origin + distance * direction;
	Math::Vector3D e01 = triangle[1] - triangle[0];
	Math::Vector3D e12 = triangle[2] - triangle[1];
	Math::Vector3D e20 = triangle[0] - triangle[2];
	Math::Vector3D e0p = p - triangle[0];
	Math::Vector3D e1p = p - triangle[1];
	Math::Vector3D e2p = p - triangle[2];
	
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

float Renderer::_intersectPlane(const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle, const Math::Vector3D &normal)
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

float Renderer::_traceRay(const Math::Vector3D &direction, const Math::Vector3D &origin, IntersectionInfo &intersection)
{
	float returnValue = 0.0f;
	float planeDistance = 0.0f;
	// FIXME Handle this in a better way
	std::vector<std::tuple<Triangle *, float, Math::Vector3D, AbstractMesh *>> planeDistances;
	
	// Intersect planes
	for (AbstractMesh *mesh : this->_meshes)
	{
		for (Triangle &triangle : mesh->triangles())
		{
			Math::Vector3D normal = triangle.normal();
			planeDistance = this->_intersectPlane(direction, origin, triangle, normal);
			
			if (planeDistance > _epsilon)
			{
				planeDistances.push_back({&triangle, planeDistance, normal, mesh});
			}
		}
	}
	
	std::sort(planeDistances.begin(), planeDistances.end(),
		[](const std::tuple<Triangle *, float, Math::Vector3D, AbstractMesh *> &left,
		   const std::tuple<Triangle *, float, Math::Vector3D, AbstractMesh *> &right)
		{
			return std::get<1>(left) < std::get<1>(right);
		}
	);
	
	// Intersect triangles
	for (decltype (planeDistances)::value_type element : planeDistances)
	{
		Triangle &currentTriangle = *std::get<0>(element);
		float distance = std::get<1>(element);
		Math::Vector3D &normal = std::get<2>(element);
		AbstractMesh *mesh = std::get<3>(element);
		
		if (this->_intersectTriangle(distance, direction, origin, currentTriangle, normal))
		{
			returnValue = distance;
			intersection.mesh = mesh;
			intersection.triangle = &currentTriangle;
			break;
		}
	}
	
	return returnValue;
}

Math::Vector3D Renderer::_castRay(const Math::Vector3D &direction, const Math::Vector3D &origin, const size_t bounce, const size_t maxBounces)
{
	Math::Vector3D returnValue = {0.0f, 0.0f, 0.0f};
	Math::Vector3D mask = {1.0f, 1.0f, 1.0f};
	
	Math::Vector3D currentDirection = direction;
	Math::Vector3D currentOrigin = origin;
	
	// FIXME This won't stay constant
	float pdf = 1.0f / (2.0f * float(M_PI));
	
	std::random_device device;
	std::default_random_engine generator(device());
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	Math::Vector3D Nt;
	Math::Vector3D Nb;
	
	for (size_t currentBounce = 0; currentBounce < maxBounces; currentBounce++)
	{
		Math::Vector3D intersectionPoint;
		IntersectionInfo intersection;
		Math::Vector3D normal;
		float distance = this->_traceRay(currentDirection, currentOrigin, intersection);
		intersectionPoint = currentOrigin + distance * currentDirection;
		
		if (intersection.triangle != nullptr)
		{
			Math::Vector3D color = intersection.mesh->material().color();
			Math::Vector3D directLight = {0.0f, 0.0f, 0.0f};
			normal = intersection.triangle->normal();
			
			// Intersection found
			for (const PointLight &pointLight : this->_pointLights)
			{
				IntersectionInfo occlusionIntersection;
				Math::Vector3D lightDirection = pointLight.position() - intersectionPoint;
				float occlusionDistance = this->_traceRay(lightDirection.normalized(), intersectionPoint, occlusionIntersection);
				
				directLight += ((normal * lightDirection.normalized()) * pointLight.color()) * 
						((occlusionIntersection.triangle == nullptr | occlusionDistance > lightDirection.magnitude()) & ((normal * lightDirection) > 0.0f));
			}
			
			directLight = directLight.coordinateProduct(color);
			
			// Indirect lighting
			this->_createCoordinateSystem(normal, Nt, Nb);
			
			// Generate hemisphere
			float r1 = distribution(generator);
			float r2 = distribution(generator);
			float sinTheta = std::pow((1.0f - r1 * r1), 0.5f);
			float phi = 2.0f * float(M_PI) * r2;
			float x = sinTheta * std::cos(phi);
			float z = sinTheta * std::sin(phi);
			Math::Vector3D sampleHemisphere{x, r1, z};
			
			Math::Matrix3D matrix{
				{Nb.x(), normal.x(), Nt.x()},
				{Nb.y(), normal.y(), Nt.y()},
				{Nb.z(), normal.z(), Nt.z()}
			};
			
			Math::Vector3D sampleWorld = matrix * sampleHemisphere;
			
			currentDirection = (intersectionPoint + sampleWorld).normalized();
			currentOrigin = sampleWorld;
			
			returnValue += mask.coordinateProduct(directLight);
			mask = mask.coordinateProduct(color);
		}
		else
		{
			break;
		}
	}
	
	return returnValue;
}

void Renderer::_createCoordinateSystem(const Math::Vector3D &N, Math::Vector3D &Nt, Math::Vector3D &Nb)
{
	if (std::abs(N.x()) > std::fabs(N.y()))
	{
		Nt = Math::Vector3D{N.z(), 0.0f, -N.x()}.normalized();
	}
	else
	{
		Nt = Math::Vector3D{0.0f, -N.z(), N.y()}.normalized();
	}
	
	Nb = N.crossProduct(Nt);
}

}
