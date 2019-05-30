#include "pointlight.h"

namespace Rendering
{

PointLight::PointLight(const Math::Vector3D &position, const Math::Vector3D &color) :
	_position(position),
	_color(color)
{
}

void PointLight::setPosition(const Math::Vector3D &position)
{
	this->_position = position;
}

const Math::Vector3D &PointLight::position() const
{
	return this->_position;
}

void PointLight::setColor(const Math::Vector3D &color)
{
	this->_color = color;
}

const Math::Vector3D &PointLight::color() const
{
	return this->_color;
}

}
