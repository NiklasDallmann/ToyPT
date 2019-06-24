#ifndef POINTLIGHT_H
#define POINTLIGHT_H

#include <vector4.h>

namespace Rendering
{

class PointLight
{
public:
	PointLight(const Math::Vector4 &position, const Math::Vector4 &color = {1, 1, 1});
	
	void setPosition(const Math::Vector4 &position);
	const Math::Vector4 &position() const;
	
	void setColor(const Math::Vector4 &color);
	const Math::Vector4 &color() const;
	
private:
	Math::Vector4 _position;
	Math::Vector4 _color;
};

} // namespace Rendering

#endif // POINTLIGHT_H
