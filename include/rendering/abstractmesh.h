#ifndef ABSTRACTMESH_H
#define ABSTRACTMESH_H

#include <matrix3d.h>
#include <vector>

#include "material.h"
#include "triangle.h"

namespace Rendering
{

class AbstractMesh
{
public:
	AbstractMesh(const Material &material = {});
	virtual ~AbstractMesh();
	
	virtual std::vector<Triangle> &triangles();
	virtual const std::vector<Triangle> &triangles() const;
	
	void setMaterial(const Material &material);
	const Material &material() const;
	
	void transform(const Math::Matrix3D &matrix);
	
	void translate(const Math::Vector3D &vector);
	
	void invert();
	
protected:
	std::vector<Triangle> _triangles;
	Material _material;
};

} // namespace Rendering

#endif // ABSTRACTMESH_H
