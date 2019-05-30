#ifndef ABSTRACTMESH_H
#define ABSTRACTMESH_H

#include <vector>

#include "triangle.h"

namespace Rendering
{

class AbstractMesh
{
public:
	AbstractMesh();
	virtual ~AbstractMesh();
	
	virtual std::vector<Triangle> &triangles() = 0;
	virtual const std::vector<Triangle> &triangles() const = 0;
	
private:
	std::vector<Triangle> _triangles;
};

} // namespace Rendering

#endif // ABSTRACTMESH_H
