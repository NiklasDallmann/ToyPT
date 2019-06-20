#include "square.h"

namespace Rendering
{

Square::Square(const size_t sideLength, const Material &material) : AbstractMesh(material)
{
	float halfSideLength = sideLength / 2.0f;
	Math::Vector3D v0, v1, v2, v3;
	
	v0 = {-halfSideLength, 0, halfSideLength};
	v1 = {halfSideLength, 0, halfSideLength};
	v2 = {halfSideLength, 0, -halfSideLength};
	v3 = {-halfSideLength, 0, -halfSideLength};
	
	this->_triangles = {
		{{v0, v1, v2}, this->_material},
		{{v0, v2, v3}, this->_material}
	};
}

}
