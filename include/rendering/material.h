#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector4.h>

namespace Rendering
{

class Material
{
public:
	Material(const Math::Vector4 &color = {}, const float roughness = 1.0f, const float metallic = 0.0f, const float specular = 0.0f);
	
	const Math::Vector4 &color() const;
	float roughness() const;
	float metallic() const;
	float specular() const;
	
private:
	Math::Vector4 _color;
	float _roughness;
	float _metallic;
	float _specular;
};

} // namespace Rendering

#endif // MATERIAL_H
