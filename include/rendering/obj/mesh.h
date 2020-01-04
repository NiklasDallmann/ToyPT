#ifndef MESH_H
#define MESH_H

#include <math/matrix4x4.h>
#include <vector>

#include "material.h"
#include "triangle.h"

namespace ToyPT::Rendering::Obj
{

// Forward declare to avoid ring include
class GeometryContainer;

class Mesh
{
public:
	Mesh(const uint32_t triangleOffset = 0, const uint32_t triangleCount = 0, const uint32_t materialOffset = 0,
		 const uint32_t vertexOffset = 0, const uint32_t vertexCount = 0, const uint32_t normalOffset = 0, const uint32_t normalCount = 0);
	
	void transform(const Math::Matrix4x4 &matrix, GeometryContainer &container);
	
	void translate(const Math::Vector4 &vector, GeometryContainer &container);
	
	void invert(GeometryContainer &container);
	
	static Mesh cube(const float sideLength, const uint32_t materialOffset,
					 GeometryContainer &container);
	
	static Mesh plane(const float sideLength, const uint32_t materialOffset,
					  GeometryContainer &container);
	
	static Mesh sphere(const float radius, const uint32_t horizontalSubDivisions, const uint32_t verticalSubDivisions, const uint32_t materialOffset,
					   GeometryContainer &container);
	
	static Math::Vector4 sphericalToCartesian(const float horizontal, const float vertical, const float radius);
	
	uint32_t triangleOffset;
	uint32_t triangleCount;
	uint32_t materialOffset;
	uint32_t vertexOffset;
	uint32_t vertexCount;
	uint32_t normalOffset;
	uint32_t normalCount;
	
};

} // namespace ToyPT::Rendering::Obj

#endif // MESH_H
