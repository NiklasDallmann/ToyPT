#include "color.h"

#include <algorithm>

namespace ToyPT::Rendering
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

uint32_t Color::rgba() const
{
	return this->_data;
}

Color Color::fromVector4(Math::Vector4 vector)
{
	Color returnValue;
	Math::Vector4 excessLight;
	size_t notExcessChannels = 0;
	uint8_t red, green, blue;
	
	// Normalize color channels
	for (size_t channel = 0; channel < 3; channel++)
	{
		const bool excess = (vector[channel] > 1.0f);
		const bool notExcess = (vector[channel] <= 1.0f);
		excessLight[channel] = excess * (vector[channel] - 1.0f);
		notExcessChannels += notExcess;
	}
	
	for (size_t channel = 0; channel < 3; channel++)
	{
		vector[(channel + 1) % 3] += excessLight[channel] / 2;
		vector[(channel + 2) % 3] += excessLight[channel] / 2;
	}
	
//	if (notExcessChannels > 0)
//	{
//		for (size_t channel = 0; channel < 3; channel++)
//		{
//			const bool notExcess = (vector[channel] <= 1.0f);
//			vector[channel] += notExcess * (excessLight / notExcessChannels);
//		}
//	}
	
	red = uint8_t(255 * std::max(0.0f, std::min(1.0f, vector.x())));
	green = uint8_t(255 * std::max(0.0f, std::min(1.0f, vector.y())));
	blue = uint8_t(255 * std::max(0.0f, std::min(1.0f, vector.z())));
	
	returnValue = {red, green, blue};
	
	return returnValue;
}

}
