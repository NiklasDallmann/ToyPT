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
	
	Rendering::Renderer renderer;
	
	Rendering::Material red{{1, 0, 0}};
	Rendering::Material green{{0, 1, 0}};
	Rendering::Material blue{{0, 0, 0.5}};
	Rendering::Material cyan{{0, 1, 1}};
	Rendering::Material magenta{{1, 0, 1}};
	Rendering::Material yellow{{1, 1, 0}};
	Rendering::Material black{{0, 0, 0}};
	Rendering::Material white{{1, 1, 1}};
	Rendering::Material halfGrey{{0.5f, 0.5f, 0.5f}};
	Rendering::Material grey{{0.2f, 0.2f, 0.2f}};
	
	renderer.materialBuffer = {red, green, blue, cyan, magenta, yellow, black, white, halfGrey, grey};
	
	Rendering::Mesh cube0 = Rendering::Mesh::cube(1, 3, renderer.triangleBuffer, renderer.vertexBuffer, renderer.normalBuffer);
	cube0.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 4.0f), renderer.vertexBuffer.data(), renderer.normalBuffer.data());
	cube0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f), renderer.vertexBuffer.data(), renderer.normalBuffer.data());
	cube0.translate({-1.5f, -0.2f, -4.5f}, renderer.vertexBuffer.data());
	
	Rendering::Mesh cube1 = Rendering::Mesh::cube(1, 4, renderer.triangleBuffer, renderer.vertexBuffer, renderer.normalBuffer);
	cube1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / -4.0f), renderer.vertexBuffer.data(), renderer.normalBuffer.data());
	cube1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / -4.0f), renderer.vertexBuffer.data(), renderer.normalBuffer.data());
	cube1.translate({2.5f, 0.2f, -5.5f}, renderer.vertexBuffer.data());
	
	Rendering::Mesh sphere = Rendering::Mesh::sphere(1.0f, 16, 8, 2, renderer.triangleBuffer, renderer.vertexBuffer, renderer.normalBuffer);
	sphere.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 4.0f), renderer.vertexBuffer.data(), renderer.normalBuffer.data());
	sphere.translate({0.0f, 0.2f, -5.0f}, renderer.vertexBuffer.data());
	
	Rendering::Mesh worldCube = Rendering::Mesh::cube(32, 9, renderer.triangleBuffer, renderer.vertexBuffer, renderer.normalBuffer);
	worldCube.invert(renderer.triangleBuffer.data(), renderer.normalBuffer.data());
	worldCube.translate({-12.0f, 15.0f, 5.0f}, renderer.vertexBuffer.data());
	
	renderer.meshBuffer.push_back(cube0);
	renderer.meshBuffer.push_back(cube1);
	renderer.meshBuffer.push_back(sphere);
	renderer.meshBuffer.push_back(worldCube);
	
	renderer.pointLightBuffer.push_back({Math::Vector4{-3.0f, 4.0f, 0.0f}, Math::Vector4{1.0f, 1.0f, 1.0f}});
	
	Rendering::FrameBuffer frameBuffer(300, 150);
	renderer.render(frameBuffer, 70, 4, 3);
	
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
