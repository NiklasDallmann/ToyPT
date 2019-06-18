#include <iostream>

#include <cube.h>
#include <framebuffer.h>
#include <matrix3d.h>
#include <renderer.h>
#include <triangle.h>
#include <vector>
#include <vector3d.h>

int main()
{
	std::cout << "Path Tracer" << std::endl;
	
	Rendering::Material red{{1, 0, 0}};
	Rendering::Material green{{0, 1, 0}};
	Rendering::Material blue{{0, 0, 1}};
	Rendering::Material cyan{{0, 1, 1}};
	Rendering::Material magenta{{1, 0, 1}};
	Rendering::Material yellow{{1, 1, 0}};
	Rendering::Material black{{0, 0, 0}};
	Rendering::Material white{{1, 1, 1}};
	Rendering::Material grey{{0.5, 0.5, 0.5}};
	
	Rendering::Cube cube(1, grey);
	cube.translate({0.0f, 0.2f, -5.0f});
	
	std::vector<Rendering::Triangle> triangles;
	triangles.push_back({{Math::Vector3D{-2, -1, -7}, Math::Vector3D{-2, -1, -3}, Math::Vector3D{2, -1, -3}}, white});
	triangles.push_back({{Math::Vector3D{-2, -1, -7}, Math::Vector3D{2, -1, -3}, Math::Vector3D{2, -1, -7}}, white});
	triangles.push_back({{Math::Vector3D{-2, -1, -7}, Math::Vector3D{2, -1, -7}, Math::Vector3D{0, 3, -7}}, blue});
	triangles.push_back({{Math::Vector3D{2, -1, -5}, Math::Vector3D{2, -1, -3}, Math::Vector3D{2, 3, -4}}, red});
	
	for (auto triangle : cube.triangles())
	{
		triangles.push_back(triangle);
	}
	
	std::vector<Rendering::PointLight> pointLights;
	pointLights.push_back({Math::Vector3D{1, 6, -10}, Math::Vector3D{1, 0, 0}});
	pointLights.push_back({Math::Vector3D{-1, 6, -10}, Math::Vector3D{0, 0, 1}});
	pointLights.push_back({Math::Vector3D{-2, 6, -10}, Math::Vector3D{0.5, 0.5, 0.5}});
	
	Rendering::FrameBuffer frameBuffer(200, 200);
	Rendering::Renderer renderer;
	renderer.setTriangles(triangles);
	renderer.setPointLights(pointLights);
	renderer.render(frameBuffer, 70, 256, 2);
	
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
