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
	Rendering::Material green{{0, 1, 0}, 1.0f};
	Rendering::Material blue{{0, 0, 0.5}, 0.0f};
	Rendering::Material cyan{{0, 0.5f, 0.5f}};
	Rendering::Material magenta{{0.5f, 0, 0.5f}};
	Rendering::Material yellow{{1, 1, 0}};
	Rendering::Material black{{0, 0, 0}};
	Rendering::Material halfWhite{{1, 1, 1}, 1.0f};
	Rendering::Material white{{1, 1, 1}, 8.0f};
	Rendering::Material halfGrey{{0.9f, 0.9f, 0.9f}};
	Rendering::Material grey{{0.2f, 0.2f, 0.2f}};
	
	renderer.geometry.materialBuffer = {red, green, blue, cyan, magenta, yellow, black, white, halfGrey, grey};
	
	Rendering::Obj::Mesh cube0 = Rendering::Obj::Mesh::cube(1, 3, renderer.geometry);
	cube0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f), renderer.geometry);
	cube0.translate({-1.5f, -0.5f, -4.5f}, renderer.geometry);
	
	Rendering::Obj::Mesh cube1 = Rendering::Obj::Mesh::cube(1, 4, renderer.geometry);
	cube1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / -4.0f), renderer.geometry);
	cube1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / -4.0f), renderer.geometry);
	cube1.translate({2.5f, 0.2f, -5.5f}, renderer.geometry);
	
//	Rendering::Obj::Mesh sphere = Rendering::Obj::Mesh::sphere(1.0f, 16, 8, 2, renderer.triangleBuffer, renderer.vertexBuffer, renderer.normalBuffer);
//	sphere.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 4.0f), renderer.vertexBuffer.data(), renderer.normalBuffer.data());
//	sphere.translate({0.0f, 0.2f, -5.0f}, renderer.vertexBuffer.data());
	
	Rendering::Obj::Mesh lightPlane0 = Rendering::Obj::Mesh::plane(1.0f, 7, renderer.geometry);
	lightPlane0.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 2.0f), renderer.geometry);
	lightPlane0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f), renderer.geometry);
	lightPlane0.translate({-2.5f, 0.0f, -5.0f}, renderer.geometry);
	
	Rendering::Obj::Mesh lightPlane1 = Rendering::Obj::Mesh::plane(2.0f, 7, renderer.geometry);
	lightPlane1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 2.0f), renderer.geometry);
	lightPlane1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 1.0f), renderer.geometry);
	lightPlane1.translate({0.0f, 0.0f, -1.0f}, renderer.geometry);
	
	Rendering::Obj::Mesh worldCube = Rendering::Obj::Mesh::cube(20, 8, renderer.geometry);
	worldCube.invert(renderer.geometry);
	worldCube.translate({-2.0f, 9.0f, -2.0f}, renderer.geometry);
	
	renderer.geometry.meshBuffer.push_back(cube0);
	renderer.geometry.meshBuffer.push_back(cube1);
//	renderer.geometry.meshBuffer.push_back(sphere);
	renderer.geometry.meshBuffer.push_back(lightPlane0);
	renderer.geometry.meshBuffer.push_back(lightPlane1);
	renderer.geometry.meshBuffer.push_back(worldCube);
	
	Rendering::FrameBuffer frameBuffer(400, 200);
	renderer.render(frameBuffer, 70, 16, 4);
	
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
