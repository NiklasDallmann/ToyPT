#include "framebuffer.h"

#include <fstream>

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

}
