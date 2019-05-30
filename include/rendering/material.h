#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector3d.h>

namespace Rendering
{

class Material
{
public:
	Material(const Math::Vector3D &color = {}, const double reflection = 0, const double refraction = 0);
	
	void setColor(const Math::Vector3D &color);
	const Math::Vector3D &color() const;
	
	void setReflection(const double reflection);
	double reflection() const;
	
	void setRefraction(const double refraction);
	double refraction() const;
	
private:
	Math::Vector3D _color;
	double _reflection;
	double _refraction;
};

} // namespace Rendering

#endif // MATERIAL_H
