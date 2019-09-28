#ifndef GEOMETRYCONTAINER_H
#define GEOMETRYCONTAINER_H

#include <vector>

#include <math/vector4.h>

#include "material.h"
#include "mesh.h"
#include "triangle.h"

namespace Rendering::Obj
{

class GeometryContainer
{
public:
	std::vector<Vertex> vertexBuffer;
	std::vector<Math::Vector4> uvBuffer;
	std::vector<Math::Vector4> normalBuffer;
	std::vector<Triangle> triangleBuffer;
	std::vector<Material> materialBuffer;
	std::vector<Mesh> meshBuffer;
};

} // namespace Rendering::Obj

#endif // GEOMETRYCONTAINER_H
