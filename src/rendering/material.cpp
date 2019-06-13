#include "material.h"

namespace Rendering
{

Material::Material(const Math::Vector3D &color, const float reflection, const float refraction) :
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

void Material::setReflection(const float reflection)
{
	this->_reflection = reflection;
}

float Material::reflection() const
{
	return this->_reflection;
}

void Material::setRefraction(const float refraction)
{
	this->_refraction = refraction;
}

float Material::refraction() const
{
	return this->_refraction;
}

}
