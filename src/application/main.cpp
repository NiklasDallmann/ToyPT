#include <iostream>

#include <algorithm>
#include <framebuffer.h>
#include <math.h>
#include <matrix4x4.h>
#include <mesh.h>
#include <renderer.h>
#include <triangle.h>
#include <vector>
#include <vector4.h>

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
	Rendering::Material halfGrey{{0.5f, 0.5f, 0.5f}};
	Rendering::Material grey{{0.2f, 0.2f, 0.2f}};
	
	Rendering::Mesh cube0 = Rendering::Mesh::cube(1, cyan);
	cube0.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 4.0f));
	cube0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f));
	cube0.translate({-1.5f, -0.2f, -4.5f});
	
	Rendering::Mesh cube1 = Rendering::Mesh::cube(1, magenta);
	cube1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / -4.0f));
	cube1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / -4.0f));
	cube1.translate({2.5f, 0.2f, -5.5f});
	
	Rendering::Mesh sphere = Rendering::Mesh::sphere(1.0f, 16, 8, white);
	sphere.translate({0.0f, 0.2f, -5.0f});
	
	Rendering::Mesh worldCube = Rendering::Mesh::cube(32, grey);
	worldCube.invert();
	worldCube.translate({-12.0f, 15.0f, 5.0f});
	
	std::vector<Rendering::Mesh> meshes;
	meshes.push_back(cube0);
	meshes.push_back(cube1);
	meshes.push_back(sphere);
	meshes.push_back(worldCube);
	
	std::vector<Rendering::PointLight> pointLights;
	pointLights.push_back({Math::Vector4{-3.0f, 4.0f, -8.0f}, Math::Vector4{1.0f, 1.0f, 1.0f}});
	
	Rendering::FrameBuffer frameBuffer(400, 200);
	Rendering::Renderer renderer;
	renderer.setMeshes(meshes);
	renderer.setPointLights(pointLights);
	renderer.render(frameBuffer, 70, 16, 3);
	
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
