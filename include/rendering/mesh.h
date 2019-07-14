#ifndef ABSTRACTMESH_H
#define ABSTRACTMESH_H

#include <matrix4x4.h>
#include <vector>

#include "material.h"
#include "triangle.h"

namespace Rendering
{

class Mesh
{
public:
	Mesh(const Material &material = {});
	
	std::vector<Triangle> &triangles();
	const std::vector<Triangle> &triangles() const;
	
	void setMaterial(const Material &material);
	const Material &material() const;
	
	void transform(const Math::Matrix4x4 &matrix);
	
	void translate(const Math::Vector4 &vector);
	
	void invert();
	
	static Mesh cube(const float sideLength, const Material &material);
	static Mesh plane(const float sideLength, const Material &material);
	static Mesh sphere(const float radius, const size_t horizontalSubDivisions, const size_t verticalSubDivisions, const Material &material);
	
protected:
	std::vector<Triangle> _triangles;
	Material _material;
	
	static Math::Vector4 _sphericalToCartesian(const float horizontal, const float vertical, const float radius);
};

} // namespace Rendering

#endif // ABSTRACTMESH_H
