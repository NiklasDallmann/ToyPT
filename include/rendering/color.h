#ifndef COLOR_H
#define COLOR_H

#include <array>
#include <stddef.h>

#include <vector3d.h>

namespace Rendering
{

class Color
{
public:
	Color(const uint8_t red = 0, const uint8_t green = 0, const uint8_t blue = 0);
	Color(const uint32_t rgba);
	
	void setRed(const uint8_t red);
	uint8_t red() const;
	
	void setGreen(const uint8_t green);
	uint8_t green() const;
	
	void setBlue(const uint8_t blue);
	uint8_t blue() const;
	
	static Color fromVector3D(const Math::Vector3D &vector);
	
private:
	uint32_t _data;
};

} // namespace Rendering

#endif // COLOR_H
