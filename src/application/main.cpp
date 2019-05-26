#include <iostream>

#include <framebuffer.h>
#include <matrix3d.h>
#include <renderer.h>
#include <triangle.h>
#include <vector>
#include <vector3d.h>

int main()
{
	std::cout << "Path Tracer" << std::endl;
	
	Rendering::Material blue{{0, 0, 1}};
	Rendering::Material red{{1, 0, 0}};
	
	std::vector<Rendering::Triangle> triangles;
	triangles.push_back({{Math::Vector3D{-0.5, 0, 2}, Math::Vector3D{0.5, -0.5, 1}, Math::Vector3D{0.5, 0.5, 1}}, red});
	triangles.push_back({{Math::Vector3D{-0.5, 0.5, 1}, Math::Vector3D{-0.5, -0.5, 1}, Math::Vector3D{0.5, 0, 2}}, blue});
//	triangles.push_back({{Math::Vector3D{0, 0.5, 1}, Math::Vector3D{-0.5, -0.5, 1}, Math::Vector3D{0.5, -0.5, 1.5}}, blue});
	
	Rendering::FrameBuffer frameBuffer(1000, 1000);
	Rendering::Renderer renderer;
	renderer.setTriangles(triangles);
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
	std::cout << "Finished." << std::endl;
	
	return 0;
}
