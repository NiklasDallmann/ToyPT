#include "color.h"

#include <algorithm>

namespace Rendering
{

Color::Color(const uint8_t red, const uint8_t green, const uint8_t blue) :
	_data(uint32_t(red) << 24 | uint32_t(green) << 16 | uint32_t(blue) << 8 | 0xff)
{
}

void Color::setRed(const uint8_t red)
{
	this->_data &= 0x00ffffff;
	this->_data |= uint32_t(red) << 24;
}

uint8_t Color::red() const
{
	return uint8_t(this->_data >> 24);
}

void Color::setGreen(const uint8_t green)
{
	this->_data &= 0xff00ffff;
	this->_data |= uint32_t(green) << 16;
}

uint8_t Color::green() const
{
	return uint8_t(this->_data >> 16);
}

void Color::setBlue(const uint8_t blue)
{
	this->_data &= 0xffff00ff;
	this->_data |= uint32_t(blue) << 8;
}

uint8_t Color::blue() const
{
	return uint8_t(this->_data >> 8);
}

Color Color::fromVector3D(const Math::Vector4 &vector)
{
	Color returnValue;
	uint8_t red, green, blue;
	
	red = uint8_t(255 * std::max(0.0f, std::min(1.0f, vector.x())));
	green = uint8_t(255 * std::max(0.0f, std::min(1.0f, vector.y())));
	blue = uint8_t(255 * std::max(0.0f, std::min(1.0f, vector.z())));
	
	returnValue = {red, green, blue};
	
	return returnValue;
}

}
