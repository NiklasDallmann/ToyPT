#include "framebuffer.h"

#include <fstream>

#include "color.h"

namespace Rendering
{

FrameBuffer::FrameBuffer(const size_t width, const size_t height) :
	_width(width),
	_height(height)
{
	this->_buffer.resize(width * height);
}

size_t FrameBuffer::width() const
{
	return this->_width;
}

size_t FrameBuffer::height() const
{
	return this->_height;
}

Math::Vector3D &FrameBuffer::pixel(const size_t x, const size_t y)
{
	return this->_buffer[x + this->_width * y];
}

bool FrameBuffer::save(const std::string &fileName)
{
	std::ofstream stream;
	stream.open(fileName);
	
	if (stream.is_open())
	{
		stream << "P6\n" << this->_width << " " << this->_height << "\n255\n";
		
		for (const Math::Vector3D &vector : this->_buffer)
		{
			Color color = Color::fromVector3D(vector);
			
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

}
