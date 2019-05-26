#include "material.h"

namespace Rendering
{

Material::Material(const Math::Vector3D &color, const Math::Vector3D &reflection, const Math::Vector3D &refraction) :
	_color(color),
	_reflection(reflection),
	_refraction(refraction)
{
}

const Math::Vector3D &Material::color() const
{
	return this->_color;
}

const Math::Vector3D &Material::reflection() const
{
	return this->_reflection;
}

const Math::Vector3D &Material::refraction() const
{
	return this->_refraction;
}

}
