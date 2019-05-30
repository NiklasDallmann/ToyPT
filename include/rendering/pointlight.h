#ifndef POINTLIGHT_H
#define POINTLIGHT_H

#include <vector3d.h>

namespace Rendering
{

class PointLight
{
public:
	PointLight(const Math::Vector3D &position, const Math::Vector3D &color = {1, 1, 1});
	
	void setPosition(const Math::Vector3D &position);
	const Math::Vector3D &position() const;
	
	void setColor(const Math::Vector3D &color);
	const Math::Vector3D &color() const;
	
private:
	Math::Vector3D _position;
	Math::Vector3D _color;
};

} // namespace Rendering

#endif // POINTLIGHT_H
