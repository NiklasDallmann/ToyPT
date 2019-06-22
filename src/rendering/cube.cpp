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
}

Cube::~Cube()
{
}

}
