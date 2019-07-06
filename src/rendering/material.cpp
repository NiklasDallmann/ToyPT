#include "material.h"

namespace Rendering
{

Material::Material(const Math::Vector4 &color, const float roughness, const float metallic, const float cavity) :
	_color(color),
	_roughness(roughness),
	_metallic(metallic),
	_cavity(cavity)
{
}

const Math::Vector4 &Material::color() const
{
	return this->_color;
}

float Material::roughness() const
{
	return this->_roughness;
}

float Material::metallic() const
{
	return this->_metallic;
}

float Material::cavity() const
{
	return this->_cavity;
}

}
