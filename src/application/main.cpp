#include <iostream>

#include <framebuffer.h>
#include <matrix3d.h>
#include <renderer.h>
#include <vector3d.h>

int main()
{
	std::cout << "Path Tracer" << std::endl;
	
	Rendering::Renderer::Triangle triangle{Math::Vector3D{0, 0.5, 1}, Math::Vector3D{-0.5, -0.5, 1}, Math::Vector3D{0.5, -0.5, 1}};
	Math::Vector3D normalVector = (triangle[1] - triangle[0]).crossProduct(triangle[2] - triangle[1]);
	
	std::cout << (triangle[1] - triangle[0]) << std::endl;
	std::cout << normalVector << std::endl;
	std::cout << triangle[0].magnitude() << std::endl;
	
	Rendering::FrameBuffer frameBuffer(100, 100);
	
//	for (size_t x = 0; x < 100; x++)
//	{
//		for (size_t y = 0; y < 100; y++)
//		{
//			double value = double(x + y) / 2.0 / 100.0;
//			Math::Vector3D vector{value, value, value};
//			frameBuffer.pixel(x, y) = vector;
//		}
//	}
	
//	std::cout << "Saving file..." << std::endl;
//	if (frameBuffer.save("img.ppm"))
//	{
//		std::cout << "Image saved." << std::endl;
//	}
//	else
//	{
//		std::cout << "Could not save image." << std::endl;
//	}
	
	Rendering::Renderer renderer;
	renderer.render(frameBuffer, 70);
	
	std::cout << "Saving file..." << std::endl;
	if (frameBuffer.save("img.ppm"))
	{
		std::cout << "Image saved." << std::endl;
	}
	else
	{
		std::cout << "Could not save image." << std::endl;
	}
	
	return 0;
}
