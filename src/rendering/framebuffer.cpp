#include "framebuffer.h"

#include <fstream>
#include <iostream>

#include <OpenImageDenoise/oidn.hpp>

#include "color.h"

namespace Rendering
{

FrameBuffer::FrameBuffer(const uint32_t width, const uint32_t height) :
	_width(width),
	_height(height)
{
	this->_buffer.resize(width * height);
}

uint32_t FrameBuffer::width() const
{
	return this->_width;
}

uint32_t FrameBuffer::height() const
{
	return this->_height;
}

Math::Vector4 &FrameBuffer::pixel(const uint32_t x, const uint32_t y)
{
	return this->_buffer[x + this->_width * y];
}

void FrameBuffer::setPixel(const uint32_t x, const uint32_t y, const Math::Vector4 &color)
{
	this->_buffer[x + this->_width * y] = color;
	this->runCallBacks(x, y);
}

bool FrameBuffer::save(const std::string &fileName)
{
	std::ofstream stream;
	stream.open(fileName);
	
	if (stream.is_open())
	{
		stream << "P6\n" << this->_width << " " << this->_height << "\n255\n";
		
		for (const Math::Vector4 &vector : this->_buffer)
		{
			Color color = Color::fromVector4(vector);
			
			stream << color.red() << color.green() << color.blue();
		}
		
		stream.close();
		
		return true;
	}
	else
	{
		return false;
	}
}

void FrameBuffer::registerCallBack(const FrameBuffer::CallBack callBack)
{
	this->_callbacks.push_back(callBack);
}

void FrameBuffer::runCallBacks(const uint32_t x, const uint32_t y)
{
	for (CallBack &callBack : this->_callbacks)
	{
		callBack(x, y);
	}
}

FrameBuffer FrameBuffer::denoise()
{
	FrameBuffer returnValue(this->_width, this->_height);
	
	
	oidn::DeviceRef device = oidn::newDevice();
	device.commit();
	
	size_t bufferSize = this->_width * this->_height * 3;
	std::vector<float> input(bufferSize);
	std::vector<float> output(bufferSize);
	
	size_t bufferIndex = 0;
	for (uint32_t h = 0; h < this->_height; h++)
	{
		for (uint32_t w = 0; w < this->_width; w++)
		{
			Math::Vector4 &pixel = this->pixel(w, h);
			
			input[bufferIndex] =		pixel.x();
			input[bufferIndex + 1] =	pixel.y();
			input[bufferIndex + 2] =	pixel.z();
			
			bufferIndex += 3;
		}
	}
	
	oidn::FilterRef filter = device.newFilter("RT");
	filter.setImage("color", input.data(), oidn::Format::Float3, this->_width, this->_height);
	filter.setImage("output", output.data(), oidn::Format::Float3, this->_width, this->_height);
	filter.commit();
	filter.execute();
	
	bufferIndex = 0;
	for (uint32_t h = 0; h < this->_height; h++)
	{
		for (uint32_t w = 0; w < this->_width; w++)
		{
			Math::Vector4 &pixel = returnValue.pixel(w, h);
			
			pixel.setX(input[bufferIndex]);
			pixel.setY(input[bufferIndex + 1]);
			pixel.setZ(input[bufferIndex + 2]);
			
			bufferIndex += 3;
		}
	}
	
	const char *errorMessage;
	if (device.getError(errorMessage) != oidn::Error::None)
	{
		std::cout << "Error: " << errorMessage << std::endl;
	}
	
	return returnValue;
}

}
