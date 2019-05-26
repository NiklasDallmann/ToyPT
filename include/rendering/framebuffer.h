#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "vector3d.h"

namespace Rendering
{

class FrameBuffer
{
public:
	FrameBuffer(const size_t width = 0, const size_t height = 0);
	
	size_t width() const;
	size_t height() const;
	
	Math::Vector3D &pixel(const size_t x, const size_t y);
	bool save(const std::string &fileName);
	
public:
	size_t _width = 0;
	size_t _height = 0;
	std::vector<Math::Vector3D> _buffer;
};

} // namespace Rendering

#endif // FRAMEBUFFER_H
