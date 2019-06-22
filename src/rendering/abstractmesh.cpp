#include "abstractmesh.h"

namespace Rendering
{

AbstractMesh::AbstractMesh(const Material &material) :
	_material(material)
{
}

AbstractMesh::~AbstractMesh()
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

void AbstractMesh::transform(const Math::Matrix3D &matrix)
{
	for (Triangle &triangle : this->_triangles)
	{
		for (Math::Vector3D &vertex : triangle.vertices())
		{
			vertex = matrix * vertex;
		}
	}
}

void AbstractMesh::translate(const Math::Vector3D &vector)
{
	for (Triangle &triangle : this->_triangles)
	{
		for (Math::Vector3D &vertex : triangle.vertices())
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
