#include "material.h"

namespace Rendering
{

Material::Material(const Math::Vector3D &color, const float roughness, const float metallic, const float specular) :
	_color(color),
	_roughness(roughness),
	_metallic(metallic),
	_specular(specular)
{
}

}
