#include "abstractmesh.h"

namespace Rendering
{

AbstractMesh::AbstractMesh(const Material &material) :
	_material(material)
{
}

std::vector<Triangle> &AbstractMesh::triangles()
{
	return this->_triangles;
}

const std::vector<Triangle> &AbstractMesh::triangles() const
{
	return this->_triangles;
}

void AbstractMesh::setMaterial(const Material &material)
{
	this->_material = material;
}

const Material &AbstractMesh::material() const
{
	return this->_material;
}

void AbstractMesh::transform(const Math::Matrix4x4 &matrix)
{
	for (Triangle &triangle : this->_triangles)
	{
		for (Math::Vector4 &vertex : triangle.vertices())
		{
			vertex = matrix * vertex;
		}
	}
}

void AbstractMesh::translate(const Math::Vector4 &vector)
{
	for (Triangle &triangle : this->_triangles)
	{
		for (Math::Vector4 &vertex : triangle.vertices())
		{
			vertex += vector;
		}
	}
}

void AbstractMesh::invert()
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

}
