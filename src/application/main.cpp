#include <iostream>

#include <algorithm>
#include <cube.h>
#include <framebuffer.h>
#include <math.h>
#include <matrix4x4.h>
#include <renderer.h>
#include <square.h>
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
	Rendering::Material grey{{0.2f, 0.2f, 0.2f}, 1.0f, 0.0f, 0.5f};
	
	Rendering::Cube cube0(1, cyan);
	cube0.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 4.0f));
	cube0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f));
	cube0.translate({-1.1f, -0.2f, -4.5f});
	
	Rendering::Cube cube1(1, magenta);
	cube1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / -4.0f));
	cube1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / -4.0f));
	cube1.translate({0.7f, 0.2f, -4.8f});
	
	Rendering::Cube worldCube(32, grey);
	worldCube.invert();
	worldCube.translate({-12.0f, 15.0f, 5.0f});
	
	std::vector<Rendering::AbstractMesh *> meshes;
	meshes.push_back(&cube0);
	meshes.push_back(&cube1);
	meshes.push_back(&worldCube);
	
	std::vector<Rendering::PointLight> pointLights;
	pointLights.push_back({Math::Vector4{-3.0f, 4.0f, -8.0f}, Math::Vector4{1.0f, 1.0f, 1.0f}});
	
	Rendering::FrameBuffer frameBuffer(100, 100);
	Rendering::Renderer renderer;
	renderer.setMeshes(meshes);
	renderer.setPointLights(pointLights);
	renderer.render(frameBuffer, 70, 32, 4);
	
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
