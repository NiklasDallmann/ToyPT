#ifndef MESH_H
#define MESH_H

#include <vector>

#include "abstractmesh.h"
#include "triangle.h"

namespace Rendering
{

class Mesh : AbstractMesh
{
public:
	Mesh(const std::vector<Triangle> &triangles);
	virtual ~Mesh();
	
	virtual std::vector<Triangle> &triangles();
	virtual const std::vector<Triangle> &triangles() const;
	
private:
	std::vector<Triangle> _triangles;
};

} // namespace Rendering

#endif // MESH_H
