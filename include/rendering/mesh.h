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
	Mesh(const uint32_t triangleOffset = 0, const uint32_t triangleCount = 0, const uint32_t materialOffset = 0,
		 const uint32_t vertexOffset = 0, const uint32_t vertexCount = 0, const uint32_t normalOffset = 0, const uint32_t normalCount = 0);
	
	void transform(const Math::Matrix4x4 &matrix, Vertex *vertexBuffer, Math::Vector4 *normalBuffer);
	
	void translate(const Math::Vector4 &vector, Vertex *vertexBuffer);
	
	void invert(Triangle *triangleBuffer, Math::Vector4 *normalBuffer);
	
	static Mesh cube(const float sideLength, const uint32_t materialOffset,
					 std::vector<Triangle> &triangleBuffer, std::vector<Vertex> &vertexBuffer,std::vector<Math::Vector4> &normalBuffer);
	
	static Mesh plane(const float sideLength, const uint32_t materialOffset,
					  std::vector<Triangle> &triangleBuffer, std::vector<Vertex> &vertexBuffer, std::vector<Math::Vector4> &normalBuffer);
	
	static Mesh sphere(const float radius, const size_t horizontalSubDivisions, const size_t verticalSubDivisions, const uint32_t materialOffset,
					   std::vector<Triangle> &triangleBuffer, std::vector<Vertex> &vertexBuffer, std::vector<Math::Vector4> &normalBuffer);
	
	static Math::Vector4 sphericalToCartesian(const float horizontal, const float vertical, const float radius);
	
	uint32_t triangleOffset;
	uint32_t triangleCount;
	uint32_t materialOffset;
	uint32_t vertexOffset;
	uint32_t vertexCount;
	uint32_t normalOffset;
	uint32_t normalCount;
	
};

} // namespace Rendering

#endif // ABSTRACTMESH_H
