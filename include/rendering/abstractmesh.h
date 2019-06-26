#ifndef ABSTRACTMESH_H
#define ABSTRACTMESH_H

#include <matrix4x4.h>
#include <vector>

#include "material.h"
#include "triangle.h"

namespace Rendering
{

class AbstractMesh
{
public:
	AbstractMesh(const Material &material = {});
	~AbstractMesh();
	
	std::vector<Triangle> &triangles();
	const std::vector<Triangle> &triangles() const;
	
	void setMaterial(const Material &material);
	const Material &material() const;
	
	void transform(const Math::Matrix4x4 &matrix);
	
	void translate(const Math::Vector4 &vector);
	
	void invert();
	
protected:
	std::vector<Triangle> _triangles;
	Material _material;
};

} // namespace Rendering

#endif // ABSTRACTMESH_H
