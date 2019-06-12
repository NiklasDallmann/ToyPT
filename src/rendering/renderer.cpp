#include <algorithm>
#include <cmath>
#include <map>
#include <matrix3d.h>
#include <random>
#include <utility>
#include <vector>

#include <iostream>
#include <sstream>

#include "renderer.h"

namespace Rendering
{

Renderer::Renderer()
{
}

void Renderer::setTriangles(const std::vector<Triangle> &triangles)
{
	this->_triangles = triangles;
}

void Renderer::setPointLights(const std::vector<PointLight> &pointLights)
{
	this->_pointLights = pointLights;
}

void Renderer::render(FrameBuffer &frameBuffer, const double fieldOfView, const size_t samples, const size_t bounces)
{
	const size_t width = frameBuffer.width();
	const size_t height = frameBuffer.height();
	double fovRadians = fieldOfView / 180.0 * M_PI;
	double zCoordinate = -(width/(2.0 * std::tan(fovRadians / 2.0)));
	size_t linesFinished = 0;
	
#pragma omp parallel for
	for (size_t j = 0; j < height; j++)
	{
		for (size_t i = 0; i < width; i++)
		{
			double x = (i + 0.5) - (width / 2.0);
			double y = -(j + 0.5) + (height / 2.0);
			
			Math::Vector3D direction{x, y, zCoordinate};
			direction.normalize();
			
			frameBuffer.pixel(i, j) = this->_castRay(direction, {0, 0, 0}, samples, 0, bounces);
		}
		
#pragma omp critical
		{
			linesFinished++;
			std::stringstream stream;
			stream << (double(linesFinished) / double(height) * 100.0) << "%\n";
			std::cout << stream.str();
		}
	}
}

bool Renderer::_intersectTriangle(const double distance, const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle, const Math::Vector3D &normal)
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
	
	double conditionInsideE01 = e01.crossProduct(e0p) * normal;
	double conditionInsideE12 = e12.crossProduct(e1p) * normal;
	double conditionInsideE20 = e20.crossProduct(e2p) * normal;
	
	if (conditionInsideE01 >= 0 & conditionInsideE12 >= 0 & conditionInsideE20 >= 0)
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

double Renderer::_intersectPlane(const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle, const Math::Vector3D &normal)
{
	double returnValue = 0;
	
//	Math::Vector3D n = triangle.normal();
	double t0 = (triangle[0] - origin) * normal;
	double t1 = normal * direction;
	
	if (t1 <= _epsilon & t1 >= -_epsilon)
	{
		returnValue = -1.0;
		goto exit;
	}
	
	returnValue = t0 / t1;
	
exit:
	return returnValue;
}

double Renderer::_traceRay(const Math::Vector3D &direction, const Math::Vector3D &origin, IntersectionInfo &intersection)
{
	double returnValue = 0;
	double planeDistance = 0;
	// FIXME Handle this in a better way
	std::vector<std::tuple<Triangle *, double, Math::Vector3D>> planeDistances;
	
	// Intersect planes
	for (Triangle &triangle : this->_triangles)
	{
		Math::Vector3D normal = triangle.normal();
		planeDistance = this->_intersectPlane(direction, origin, triangle, normal);
		
		if (planeDistance > _epsilon)
		{
			planeDistances.push_back({&triangle, planeDistance, normal});
		}
	}
	
	std::sort(planeDistances.begin(), planeDistances.end(), [](const std::tuple<Triangle *, double, Math::Vector3D> &left, const std::tuple<Triangle *, double, Math::Vector3D> &right){
		return std::get<1>(left) < std::get<1>(right);
	});
	
	// Intersect triangles
	for (decltype (planeDistances)::value_type element : planeDistances)
	{
		Triangle &currentTriangle = *std::get<0>(element);
		double distance = std::get<1>(element);
		Math::Vector3D &normal = std::get<2>(element);
		
		if (this->_intersectTriangle(distance, direction, origin, currentTriangle, normal))
		{
			returnValue = distance;
			intersection.triangle = &currentTriangle;
			break;
		}
	}
	
	return returnValue;
}

Math::Vector3D Renderer::_castRay(const Math::Vector3D &direction, const Math::Vector3D &origin, const size_t samples, const size_t bounce, const size_t maxBounces)
{
	Math::Vector3D returnValue = {0, 0, 0};
	Math::Vector3D intersectionPoint;
	Math::Vector3D directLight = {0, 0, 0};
	Math::Vector3D indirectLight = {0, 0, 0};
	double distance = 0;
	IntersectionInfo intersection;
	Math::Vector3D normal;
	
	if (bounce == maxBounces)
	{
		goto exit;
	}
	
	// Direct lighting
	distance = this->_traceRay(direction, origin, intersection);
	intersectionPoint = origin + distance * direction;
	
	if (intersection.triangle != nullptr)
	{
		returnValue = intersection.triangle->material().color();
		normal = intersection.triangle->normal();
		
		// Intersection found
		for (const PointLight &pointLight : this->_pointLights)
		{
			IntersectionInfo occlusionIntersection;
			Math::Vector3D lightDirection = pointLight.position() - intersectionPoint;
			double newDistance = this->_traceRay(lightDirection.normalized(), intersectionPoint, occlusionIntersection);
			
			if ((occlusionIntersection.triangle == nullptr | newDistance > lightDirection.magnitude()) & ((normal * lightDirection) > 0))
			{
				// Point light visible
				directLight += pointLight.color();// * (1.0 / std::pow(lightDirection.magnitude() / 8, 2.0));
			}
		}
		
//		returnValue = directLight.coordinateProduct(intersection.triangle->material().color()) / 2.0 + intersection.triangle->material().color() / 2.0;
//		returnValue = directLight.coordinateProduct(intersection.triangle->material().color());
		directLight = directLight.coordinateProduct(intersection.triangle->material().color());
		
		// Indirect lighting
		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution(0, 1);
		Math::Vector3D Nt;
		Math::Vector3D Nb;
		double pdf = 1.0 / (2.0 * M_PI);
		
		this->_createCoordinateSystem(normal, Nt, Nb);
		for (size_t sample = 0; sample < samples; sample++)
		{
			// Generate hemisphere
			double r1 = distribution(generator);
			double r2 = distribution(generator);
			double sinTheta = std::pow((1.0 - r1 * r1), 0.5);
			double phi = 2.0 * M_PI * r2;
			double x = sinTheta * std::cos(phi);
			double z = sinTheta * std::sin(phi);
			Math::Vector3D sampleHemisphere{x, r1, z};
			
			Math::Vector3D sampleWorld{
				sampleHemisphere.x() * Nb.x() + sampleHemisphere.y() * normal.x() + sampleHemisphere.z() * Nt.x(),
				sampleHemisphere.x() * Nb.y() + sampleHemisphere.y() * normal.y() + sampleHemisphere.z() * Nt.y(),
				sampleHemisphere.x() * Nb.z() + sampleHemisphere.y() * normal.z() + sampleHemisphere.z() * Nt.z()
			};
			
			Math::Vector3D indirectColor = this->_castRay(sampleWorld.normalized(), intersectionPoint, samples, bounce + 1, maxBounces);
			indirectLight += r1 * indirectColor / pdf;
			
//			std::stringstream stream;
//			stream << (intersectionPoint + sampleWorld) << "\n";
//			stream << "r1=" << r1 << " r2=" << r2 << "\n";
//			std::cout << stream.str();
			
//			if (indirectLight != Math::Vector3D{0, 0, 0})
//			{
//				stream.clear();
//				stream << "indirect light " << indirectLight << "\n";
//				std::cout << stream.str();
//			}
		}
		
		indirectLight /= double(samples);
		
		returnValue = (directLight / M_PI + 2.0 * indirectLight);
//		returnValue = directLight + indirectLight;
//		returnValue = directLight;
	}
	
exit:
	return returnValue;
}

void Renderer::_createCoordinateSystem(const Math::Vector3D &N, Math::Vector3D &Nt, Math::Vector3D &Nb)
{
	if (std::abs(N.x()) > std::fabs(N.y()))
	{
		Nt = Math::Vector3D{N.z(), 0, -N.x()}.normalized();
	}
	else
	{
		Nt = Math::Vector3D{0, -N.z(), N.y()}.normalized();
	}
	
	Nb = N.crossProduct(Nt);
}

}
