#include "triangle.h"

namespace Rendering
{

Triangle::Triangle(const std::array<Math::Vector3D, 3> &vertices, const Material &material) :
	_vertices(vertices),
	_material(material)
{
}

void Triangle::setMaterial(const Material &material)
{
	this->_material = material;
}

const Material &Triangle::material() const
{
	return this->_material;
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
