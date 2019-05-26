#include <cmath>
#include <vector>

#include <iostream>

#include "renderer.h"

namespace Rendering
{

Renderer::Renderer()
{
	
}

void Renderer::render(FrameBuffer &frameBuffer, double fieldOfView)
{
	const size_t width = frameBuffer.width();
	const size_t height = frameBuffer.height();
	std::vector<Triangle> triangles;
	double fovRadians = fieldOfView / 180.0 * M_PI;
	
	Triangle triangle{Math::Vector3D{0, 0.5, 1}, Math::Vector3D{-0.5, -0.5, 1}, Math::Vector3D{0.5, -0.5, 1}};
	
	triangles.push_back(triangle);
	
	for (size_t j = 0; j < height; j++)
	{
		for (size_t i = 0; i < width; i++)
		{
			double x = (i + 0.5) - (width / 2.0);
			double y = -(j + 0.5) + (height / 2.0);
			double z = height/(2.0 * std::tan(fovRadians / 2.0));
			
			Math::Vector3D ray{x, y, z};
			ray.normalize();
			
			for (Triangle &triangle : triangles)
			{
				Math::Vector3D n = (triangle[1] - triangle[0]).crossProduct(triangle[2] - triangle[1]);
				double d = (-n[0] * -triangle[0][0]) + (-n[1] * -triangle[1][1]) + (-n[2] * -triangle[2][2]);
				double t0 = -(n * ray + d);
				double t1 = n * ray;
//				double t = t0 / t1;
//				Math::Vector3D r = t * ray;
				
				if (t1 <= 0.1 || t1 >= -0.1)
				{
					frameBuffer.pixel(i, j) = Math::Vector3D(1, 1, 1);
				}
				else
				{
					frameBuffer.pixel(i, j) = Math::Vector3D(0, 0, 0);
				}
			}
		}
	}
}

}
