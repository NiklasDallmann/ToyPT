#include "triangle.h"

#include <assert.h>

namespace Rendering
{

Triangle::Triangle(const std::array<Math::Vector4, 3> &vertices, const std::array<Math::Vector4, 3> &normals) :
	_vertices(vertices),
	_normals(normals)
{
}

Math::Vector4 Triangle::normal() const
{
	return (this->_vertices[1] - this->_vertices[0]).crossProduct(this->_vertices[2] - this->_vertices[0]).normalized();
}

std::array<Math::Vector4, 3> &Triangle::vertices()
{
	return this->_vertices;
}

const std::array<Math::Vector4, 3> &Triangle::vertices() const
{
	return this->_vertices;
}

std::array<Math::Vector4, 3> &Triangle::normals()
{
	return this->_normals;
}

const std::array<Math::Vector4, 3> &Triangle::normals() const
{
	return this->_normals;
}

Math::Vector4 &Triangle::operator[](const size_t index)
{
	return this->_vertices[index];
}

const Math::Vector4 &Triangle::operator[](const size_t index) const
{
	return this->_vertices[index];
}

}
