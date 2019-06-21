#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector3d.h>

namespace Rendering
{

class Material
{
public:
	Material(const Math::Vector3D &color = {}, const float roughness = 1.0f, const float metallic = 0.0f, const float specular = 0.0f);
	
private:
	Math::Vector3D _color;
	float _roughness;
	float _metallic;
	float _specular;
};

} // namespace Rendering

#endif // MATERIAL_H
