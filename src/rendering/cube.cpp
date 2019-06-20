#include "cube.h"

namespace Rendering
{

Cube::Cube(const float sideLength, const Material &material) : AbstractMesh(material)
{
	// Each vertex is offset by half the side length on two axes
	float halfSideLength = sideLength / 2.0f;
	
	// Create vertices, a cube has four of them
	Math::Vector3D v0, v1, v2, v3, v4, v5, v6, v7;
	
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
	this->_triangles = {
		// Upper face
		{{v0, v1, v2}, this->_material},
		{{v0, v2, v3}, this->_material},
		// Lower face
		{{v4, v5, v6}, this->_material},
		{{v4, v6, v7}, this->_material},
		// Front face
		{{v4, v5, v1}, this->_material},
		{{v4, v1, v0}, this->_material},
		// Back face
		{{v7, v6, v2}, this->_material},
		{{v7, v2, v3}, this->_material},
		// Left face
		{{v4, v0, v3}, this->_material},
		{{v4, v3, v7}, this->_material},
		// Right face
		{{v5, v6, v2}, this->_material},
		{{v5, v2, v1}, this->_material},
	};
}

Cube::~Cube()
{
}

}
