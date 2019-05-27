#include <algorithm>
#include <cmath>
#include <map>
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

void Renderer::setSun(const Math::Vector3D &sun)
{
	this->_sun = sun;
}

void Renderer::render(FrameBuffer &frameBuffer, double fieldOfView)
{
	const size_t width = frameBuffer.width();
	const size_t height = frameBuffer.height();
	double fovRadians = fieldOfView / 180.0 * M_PI;
	double zCoordinate = width/(2.0 * std::tan(fovRadians / 2.0));
	
#pragma omp parallel for
	for (size_t j = 0; j < height; j++)
	{
		for (size_t i = 0; i < width; i++)
		{
			double x = (i + 0.5) - (width / 2.0);
			double y = -(j + 0.5) + (height / 2.0);
			
			Math::Vector3D direction{x, y, zCoordinate};
			direction.normalize();
			
			frameBuffer.pixel(i, j) = this->_castRay(direction, {});
		}
	}
}

bool Renderer::_intersectTriangle(const double distance, const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle)
{
	bool returnValue = false;
	Math::Vector3D n = triangle.normal();
	Math::Vector3D p = origin + distance * direction;
	Math::Vector3D e01 = triangle[1] - triangle[0];
	Math::Vector3D e12 = triangle[2] - triangle[1];
	Math::Vector3D e20 = triangle[0] - triangle[2];
	Math::Vector3D e0p = p - triangle[0];
	Math::Vector3D e1p = p - triangle[1];
	Math::Vector3D e2p = p - triangle[2];
	
	double conditionInsideE01 = e01.crossProduct(e0p) * n;
	double conditionInsideE12 = e12.crossProduct(e1p) * n;
	double conditionInsideE20 = e20.crossProduct(e2p) * n;
	
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

double Renderer::_intersectPlane(const Math::Vector3D &direction, const Math::Vector3D &origin, const Triangle &triangle)
{
	double returnValue = 0;
	
	Math::Vector3D n = triangle.normal();
	double d = (-n[0] * triangle[0][0]) + (-n[1] * triangle[0][1]) + (-n[2] * triangle[0][2]);
	double t0 = -d + n * origin;
	double t1 = n * direction;
	
	if (t1 <= _epsilon & t1 >= -_epsilon)
	{
		returnValue = -1.0;
		goto exit;
	}
	
	returnValue = t0 / t1;
	
exit:
	return returnValue;
}

Math::Vector3D Renderer::_castRay(const Math::Vector3D &direction, const Math::Vector3D &origin, const size_t bounce)
{
	Math::Vector3D returnValue;
	double planeDistance = 0;
	std::vector<std::pair<Triangle, double>> planeDistances;
	
	// Intersect planes
	for (Triangle &triangle : this->_triangles)
	{
		planeDistance = this->_intersectPlane(direction, origin, triangle);
		
		if (planeDistance > 0)
		{
			planeDistances.push_back({triangle, planeDistance});
		}
	}
	
	std::sort(planeDistances.begin(), planeDistances.end(), [](const std::pair<Triangle, double> &left, const std::pair<Triangle, double> &right){
		return left.second < right.second;
	});
	
	// Intersect triangles
	for (decltype (planeDistances)::value_type element : planeDistances)
	{
		Triangle &triangle = element.first;
		double distance = element.second;
		
		if (this->_intersectTriangle(distance, direction, {}, triangle))
		{
			Math::Vector3D normal = triangle.normal();
			Math::Vector3D newOrigin = origin + direction * distance;
			Math::Vector3D newDirection = (2.0 * (normal * ((-1.0) * direction)) * normal + direction).normalized();
			
//			if (bounce > 0)
//			{
//				std::cout << "bounced" << std::endl;
//			}
			
			returnValue = triangle.material().color().coordinateProduct(this->_castRay(newDirection, newOrigin, bounce + 1));
			goto exit;
		}
	}
	
	returnValue = {1, 1, 1};
	
exit:
	return returnValue;
}

}
