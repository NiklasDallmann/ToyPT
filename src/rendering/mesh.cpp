#include <cmath>
#include <math.h>

#include "mesh.h"

namespace Rendering
{

Mesh::Mesh(const Material &material) :
	_material(material)
{
}

std::vector<Triangle> &Mesh::triangles()
{
	return this->_triangles;
}

const std::vector<Triangle> &Mesh::triangles() const
{
	return this->_triangles;
}

void Mesh::setMaterial(const Material &material)
{
	this->_material = material;
}

const Material &Mesh::material() const
{
	return this->_material;
}

void Mesh::transform(const Math::Matrix4x4 &matrix)
{
	for (Triangle &triangle : this->_triangles)
	{
		for (Math::Vector4 &vertex : triangle.vertices())
		{
			vertex = matrix * vertex;
		}
	}
}

void Mesh::translate(const Math::Vector4 &vector)
{
	for (Triangle &triangle : this->_triangles)
	{
		for (Math::Vector4 &vertex : triangle.vertices())
		{
			vertex += vector;
		}
	}
}

void Mesh::invert()
{
	for (Triangle &triangle : this->_triangles)
	{
		Triangle inverse{
			{
				triangle[2],
				triangle[1],
				triangle[0]
			}
		};
		
		triangle = inverse;
	}
}

Mesh Mesh::cube(const float sideLength, const Material &material)
{
	Mesh returnValue(material);
	
	// Each vertex is offset by half the side length on two axes
	float halfSideLength = sideLength / 2.0f;
	
	// Create vertices, a cube has four of them
	Math::Vector4 v0, v1, v2, v3, v4, v5, v6, v7;
	
	// Upper four
	v0 = {-halfSideLength, halfSideLength, halfSideLength};
	v1 = {halfSideLength, halfSideLength, halfSideLength};
	v2 = {halfSideLength, halfSideLength, -halfSideLength};
	v3 = {-halfSideLength, halfSideLength, -halfSideLength};
	
	// Lower four
	v4 = {-halfSideLength, -halfSideLength, halfSideLength};
	v5 = {halfSideLength, -halfSideLength, halfSideLength};
	v6 = {halfSideLength, -halfSideLength, -halfSideLength};
	v7 = {-halfSideLength, -halfSideLength, -halfSideLength};
	
	// Create Triangles
	returnValue._triangles = {
		// Upper face
		{{v0, v1, v2}},
		{{v0, v2, v3}},
		// Lower face
		{{v6, v5, v4}},
		{{v7, v6, v4}},
		// Front face
		{{v4, v5, v1}},
		{{v4, v1, v0}},
		// Back face
		{{v6, v7, v3}},
		{{v6, v3, v2}},
		// Left face
		{{v4, v0, v3}},
		{{v4, v3, v7}},
		// Right face
		{{v5, v6, v2}},
		{{v5, v2, v1}},
	};
	
	return returnValue;
}

Mesh Mesh::plane(const float sideLength, const Material &material)
{
	Mesh returnValue(material);
	
	float halfSideLength = sideLength / 2.0f;
	Math::Vector4 v0, v1, v2, v3;
	
	v0 = {-halfSideLength, 0, halfSideLength};
	v1 = {halfSideLength, 0, halfSideLength};
	v2 = {halfSideLength, 0, -halfSideLength};
	v3 = {-halfSideLength, 0, -halfSideLength};
	
	returnValue._triangles = {
		{{v0, v1, v2}},
		{{v0, v2, v3}}
	};
	
	return returnValue;
}

Mesh Mesh::sphere(const float radius, const size_t horizontalSubDivisions, const size_t verticalSubDivisions, const Material &material)
{
	Mesh returnValue(material);
//	const float verticalOffset = float(M_PI) / verticalSubDivisions;
//	const float horizontalOffset = 2.0f * float(M_PI) / horizontalSubDivisions;
	
	// Generate vertices
	for (size_t vertical = 0; vertical < verticalSubDivisions; vertical++)
	{
		const float theta1 = float(M_PI) * (float(vertical) / float(verticalSubDivisions));
		const float theta2 = float(M_PI) * (float(vertical + 1) / float(verticalSubDivisions));
		
		for (size_t horizontal = 0; horizontal < horizontalSubDivisions; horizontal++)
		{
			const float phi1 = float(M_PI) * 2.0f * (float(horizontal) / float(horizontalSubDivisions));
			const float phi2 = float(M_PI) * 2.0f * (float(horizontal + 1) / float(horizontalSubDivisions));
			
			const Math::Vector4 v1 = _sphericalToCartesian(phi1, theta1, radius);
			const Math::Vector4 v2 = _sphericalToCartesian(phi2, theta1, radius);
			const Math::Vector4 v3 = _sphericalToCartesian(phi2, theta2, radius);
			const Math::Vector4 v4 = _sphericalToCartesian(phi1, theta2, radius);
			
			if (verticalSubDivisions == 0)
			{
				returnValue._triangles.push_back({{v1, v3, v4}});
			}
			else if (verticalSubDivisions == (verticalSubDivisions - 1))
			{
				returnValue._triangles.push_back({{v3, v1, v2}});
			}
			else
			{
				returnValue._triangles.push_back({{v1, v2, v4}});
				returnValue._triangles.push_back({{v2, v3, v4}});
			}
		}
	}
	
	return returnValue;
}

Math::Vector4 Mesh::_sphericalToCartesian(const float horizontal, const float vertical, const float radius)
{
	Math::Vector4 returnValue;
	
	const float x = radius * std::sin(vertical) * std::cos(horizontal);
	const float y = radius * std::cos(vertical);
	const float z = radius * std::sin(vertical) * std::sin(horizontal);
	
	returnValue = {x, y, z};
	
	return returnValue;
}

}
