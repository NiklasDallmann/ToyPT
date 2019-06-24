#include "pointlight.h"

namespace Rendering
{

PointLight::PointLight(const Math::Vector4 &position, const Math::Vector4 &color) :
	_position(position),
	_color(color)
{
}

void PointLight::setPosition(const Math::Vector4 &position)
{
	this->_position = position;
}

const Math::Vector4 &PointLight::position() const
{
	return this->_position;
}

void PointLight::setColor(const Math::Vector4 &color)
{
	this->_color = color;
}

const Math::Vector4 &PointLight::color() const
{
	return this->_color;
}

}
