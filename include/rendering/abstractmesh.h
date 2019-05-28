#ifndef ABSTRACTMESH_H
#define ABSTRACTMESH_H


namespace Rendering
{

class AbstractMesh
{
public:
	AbstractMesh();
	virtual ~AbstractMesh();
	
	virtual std::vector<Triangle> &triangles() = 0;
	virtual const std::vector<Triangle> &triangles() const = 0;
};

} // namespace Rendering

#endif // ABSTRACTMESH_H
