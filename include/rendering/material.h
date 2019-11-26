#ifndef MATERIAL_H
#define MATERIAL_H

#include <math/vector4.h>

namespace ToyPT
{
namespace Rendering
{

class Material
{
public:
	Material(const Math::Vector4 &color = {}, const float emittance = 0.0f, const float roughness = 1.0f, const float metallic = 0.0f,
			 const float cavity = 0.04f, const Math::Vector4 reflectance = {0.5f}) :
		color(color),
		emittance(emittance),
		roughness(roughness),
		metallic(metallic),
		cavity(cavity),
		reflectance(reflectance)
	{
	}
	
	Math::Vector4	color;
	float			emittance;
	float			roughness;
	float			metallic;
	float			cavity;
	Math::Vector4	reflectance;
};

} // namespace Rendering
} // namespace ToyPT

#endif // MATERIAL_H
