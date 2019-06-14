#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <array>
#include <stddef.h>
#include <vector3d.h>

#include "material.h"

namespace Rendering
{

class Triangle
{
public:
	Triangle(const std::array<Math::Vector3D, 3> &vertices = {}, const Material &material = {});
	
	void setMaterial(const Material &material);
	const Material &material() const;
	
	Math::Vector3D normal() const;
	
	std::array<Math::Vector3D, 3> &vertices();
	const std::array<Math::Vector3D, 3> &vertices() const;
	
	Math::Vector3D &operator[](const size_t index);
	const Math::Vector3D &operator[](const size_t index) const;
	
private:
	std::array<Math::Vector3D, 3> _vertices;
	Material _material;
};

} // namespace Rendering

#endif // TRIANGLE_H
