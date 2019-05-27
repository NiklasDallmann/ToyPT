#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector3d.h>

namespace Rendering
{

class Material
{
public:
	Material(const Math::Vector3D &color = {}, const Math::Vector3D &reflection = {}, const Math::Vector3D &refraction = {});
	
	void setColor(const Math::Vector3D &color);
	const Math::Vector3D &color() const;
	
	void setRefraction(const Math::Vector3D &refraction);
	const Math::Vector3D &reflection() const;
	
	void setReflection(const Math::Vector3D &reflection);
	const Math::Vector3D &refraction() const;
	
private:
	Math::Vector3D _color;
	Math::Vector3D _reflection;
	Math::Vector3D _refraction;
};

} // namespace Rendering

#endif // MATERIAL_H
