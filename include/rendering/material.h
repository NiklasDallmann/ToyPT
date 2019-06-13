#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector3d.h>

namespace Rendering
{

class Material
{
public:
	Material(const Math::Vector3D &color = {}, const float reflection = 0, const float refraction = 0);
	
	void setColor(const Math::Vector3D &color);
	const Math::Vector3D &color() const;
	
	void setReflection(const float reflection);
	float reflection() const;
	
	void setRefraction(const float refraction);
	float refraction() const;
	
private:
	Math::Vector3D _color;
	float _reflection;
	float _refraction;
};

} // namespace Rendering

#endif // MATERIAL_H
