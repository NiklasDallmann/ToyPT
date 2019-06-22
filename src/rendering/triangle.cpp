#include "triangle.h"

#include <assert.h>

namespace Rendering
{

Triangle::Triangle(const std::array<Math::Vector3D, 3> &vertices) :
	_vertices(vertices)
{
}

Math::Vector3D Triangle::normal() const
{
	return (this->_vertices[1] - this->_vertices[0]).crossProduct(this->_vertices[2] - this->_vertices[0]).normalized();
}

std::array<Math::Vector3D, 3> &Triangle::vertices()
{
	return this->_vertices;
}

const std::array<Math::Vector3D, 3> &Triangle::vertices() const
{
	return this->_vertices;
}

Math::Vector3D &Triangle::operator[](const size_t index)
{
	return this->_vertices[index];
}

const Math::Vector3D &Triangle::operator[](const size_t index) const
{
	return this->_vertices[index];
}

}
