#include "framebuffer.h"

#include <cstring>
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

const Math::Vector4 &FrameBuffer::pixel(const uint32_t x, const uint32_t y) const
{
	return this->_buffer[x + this->_width * y];
}

void FrameBuffer::setPixel(const uint32_t x, const uint32_t y, const Math::Vector4 &color)
{
	this->_buffer[x + this->_width * y] = color;
	this->runCallBacks(x, y);
}

Math::Vector4 *FrameBuffer::data()
{
	return this->_buffer.data();
}

const Math::Vector4 *FrameBuffer::data() const
{
	return this->_buffer.data();
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

FrameBuffer FrameBuffer::denoise(const FrameBuffer &color)
{
	uint32_t width = color.width();
	uint32_t height = color.height();
	
	FrameBuffer returnValue(width, height);
	
	oidn::DeviceRef device = oidn::newDevice();
	device.commit();
	
	size_t bufferSize = width * height * 3;
	std::vector<float> colorInput(bufferSize);
	std::vector<float> albedoInput(bufferSize);
	std::vector<float> normalInput(bufferSize);
	std::vector<float> output(bufferSize);
	
	_frameBufferToBuffer(color, colorInput);
	
	oidn::FilterRef filter = device.newFilter("RT");
	filter.setImage("color", colorInput.data(), oidn::Format::Float3, width, height);
	filter.setImage("output", output.data(), oidn::Format::Float3, width, height);
	filter.set("hdr", false);
	filter.set("srgb", false);
	filter.commit();
	filter.execute();
	
	_bufferToFrameBuffer(output, returnValue);
	
	const char *errorMessage;
	if (device.getError(errorMessage) != oidn::Error::None)
	{
		std::cout << "Error: " << errorMessage << std::endl;
	}
	
	return returnValue;
}

FrameBuffer FrameBuffer::fromRawData(const Math::Vector4 *data, const uint32_t width, const uint32_t height)
{
	const uint32_t pixelCount = width * height;
	FrameBuffer returnValue(width, height);
	
	std::memcpy(returnValue.data(), data, pixelCount);
	
	return returnValue;
}

void FrameBuffer::_frameBufferToBuffer(const FrameBuffer &frameBuffer, std::vector<float> &buffer)
{
	size_t bufferIndex = 0;
	for (uint32_t h = 0; h < frameBuffer._height; h++)
	{
		for (uint32_t w = 0; w < frameBuffer._width; w++)
		{
			const Math::Vector4 pixel = frameBuffer.pixel(w, h);
			
			buffer[bufferIndex] =		pixel.x();
			buffer[bufferIndex + 1] =	pixel.y();
			buffer[bufferIndex + 2] =	pixel.z();
			
			bufferIndex += 3;
		}
	}
}

void FrameBuffer::_bufferToFrameBuffer(const std::vector<float> &buffer, FrameBuffer &frameBuffer)
{
	size_t bufferIndex = 0;
	for (uint32_t h = 0; h < frameBuffer._height; h++)
	{
		for (uint32_t w = 0; w < frameBuffer._width; w++)
		{
			Math::Vector4 &pixel = frameBuffer.pixel(w, h);
			
			pixel.setX(buffer[bufferIndex]);
			pixel.setY(buffer[bufferIndex + 1]);
			pixel.setZ(buffer[bufferIndex + 2]);
			
			bufferIndex += 3;
		}
	}
}

}
