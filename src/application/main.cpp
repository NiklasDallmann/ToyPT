#include <iostream>

#include <algorithm>
#include <cube.h>
#include <framebuffer.h>
#include <math.h>
#include <matrix3d.h>
#include <renderer.h>
#include <square.h>
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
	Rendering::Material grey{{0.2f, 0.2f, 0.2f}};
	
	Rendering::Cube cube(1, cyan);
	cube.transform(Math::Matrix3D::rotationMatrixX(0.7f));
	cube.transform(Math::Matrix3D::rotationMatrixY(0.7f));
	cube.translate({0.0f, -0.2f, -4.5f});
	
	Rendering::Square square0(32, grey);
	square0.transform(Math::Matrix3D::rotationMatrixX(float(M_PI) / 2.0f));
	square0.translate({0.0f, 10.0f, -10.0f});
	
	Rendering::Square square1(32, grey);
	square1.translate({0.0f, -1.0f, -2.0f});
	
	std::vector<Rendering::Triangle> triangles;
	triangles.insert(triangles.end(), cube.triangles().cbegin(), cube.triangles().cend());
	triangles.insert(triangles.end(), square0.triangles().cbegin(), square0.triangles().cend());
	triangles.insert(triangles.end(), square1.triangles().cbegin(), square1.triangles().cend());
	
	std::vector<Rendering::PointLight> pointLights;
	pointLights.push_back({Math::Vector3D{-2.0f, 4.0f, -10.0f}, Math::Vector3D{0.5f, 0.5f, 0.5f}});
	
	Rendering::FrameBuffer frameBuffer(200, 200);
	Rendering::Renderer renderer;
	renderer.setTriangles(triangles);
	renderer.setPointLights(pointLights);
	renderer.render(frameBuffer, 70, 64, 3);
	
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
