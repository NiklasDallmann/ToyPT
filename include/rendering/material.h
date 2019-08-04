#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector4.h>

namespace Rendering
{

class Material
{
public:
	Material(const Math::Vector4 &color = {}, const float emittance = 0.0f, const float roughness = 1.0f, const float metallic = 0.0f,
			 const float cavity = 0.04f);
	
	const Math::Vector4 &color() const;
	float emittance() const;
	float roughness() const;
	float metallic() const;
	float cavity() const;
	
private:
	Math::Vector4 _color;
	float _emittance;
	float _roughness;
	float _metallic;
	float _cavity;
};

} // namespace Rendering

#endif // MATERIAL_H
