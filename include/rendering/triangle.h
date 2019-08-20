#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <array>
#include <iostream>
#include <stddef.h>
#include <vector4.h>

namespace Rendering::Obj
{

using Vertex = Math::Vector4;

struct Triangle
{
	Triangle(const std::array<uint32_t, 3> &vertices, const std::array<uint32_t, 3> &uvCoordinates = {}, const std::array<uint32_t, 3> &normals = {}) :
		vertices(vertices),
		uvCoordinates(uvCoordinates),
		normals(normals)
	{
	}
	
	static Math::Vector4 normal(const Triangle *triangle, const Vertex *vertexBuffer)
	{
		Math::Vector4 returnValue;
		auto vertexIndices = triangle->vertices;
		
		returnValue = (vertexBuffer[vertexIndices[1]] - vertexBuffer[vertexIndices[0]]).crossProduct(
					vertexBuffer[vertexIndices[2]] - vertexBuffer[vertexIndices[0]]).normalized();
		
		return returnValue;
	}
	
	std::array<uint32_t, 3> vertices;
	std::array<uint32_t, 3> uvCoordinates;
	std::array<uint32_t, 3> normals;
};

inline std::ostream &operator<<(std::ostream &stream, const Triangle &triangle)
{
	std::stringstream stringStream;
	
	stringStream << "{" << triangle.vertices[0] << ", " << triangle.vertices[1] << ", " << triangle.vertices[2] << "}";
	
	stream << stringStream.str();
	
	return stream;
}

} // namespace Rendering::Obj

#endif // TRIANGLE_H
