#include "material.h"

namespace Rendering
{

Material::Material(const Math::Vector3D &color, const double reflection, const double refraction) :
	_color(color),
	_reflection(reflection),
	_refraction(refraction)
{
}

void Material::setColor(const Math::Vector3D &color)
{
	this->_color = color;
}

const Math::Vector3D &Material::color() const
{
	return this->_color;
}

void Material::setReflection(const double reflection)
{
	this->_reflection = reflection;
}

double Material::reflection() const
{
	return this->_reflection;
}

void Material::setRefraction(const double refraction)
{
	this->_refraction = refraction;
}

double Material::refraction() const
{
	return this->_refraction;
}

}
