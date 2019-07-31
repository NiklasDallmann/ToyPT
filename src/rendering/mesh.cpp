#include <cmath>
#include <math.h>

#include "mesh.h"

namespace Rendering
{

Mesh::Mesh(const uint32_t triangleOffset, const uint32_t triangleCount, const uint32_t materialOffset, const uint32_t vertexOffset, const uint32_t vertexCount,
		   const uint32_t normalOffset, const uint32_t normalCount) :
	triangleOffset(triangleOffset),
	triangleCount(triangleCount),
	materialOffset(materialOffset),
	vertexOffset(vertexOffset),
	vertexCount(vertexCount),
	normalOffset(normalOffset),
	normalCount(normalCount)
{
}

void Mesh::transform(const Math::Matrix4x4 &matrix, Vertex *vertexBuffer, Math::Vector4 *normalBuffer)
{
	for (uint32_t vertexIndex = this->vertexOffset; vertexIndex < (this->vertexOffset + vertexCount); vertexIndex++)
	{
		vertexBuffer[vertexIndex] = matrix * vertexBuffer[vertexIndex];
	}
	
	for (uint32_t normalIndex = this->normalOffset; normalIndex < (this->normalOffset + normalCount); normalIndex++)
	{
		normalBuffer[normalIndex] = matrix * normalBuffer[normalIndex];
	}
}

void Mesh::translate(const Math::Vector4 &vector, Vertex *vertexBuffer)
{
	for (uint32_t vertexIndex = this->vertexOffset; vertexIndex < (this->vertexOffset + this->vertexCount); vertexIndex++)
	{
		vertexBuffer[vertexIndex] += vector;
	}
}

void Mesh::invert(Triangle *triangleBuffer, Math::Vector4 *normalBuffer)
{
	for (uint32_t triangleIndex = this->triangleOffset; triangleIndex < (this->triangleOffset + this->triangleCount); triangleIndex++)
	{
		Triangle &triangle = triangleBuffer[triangleIndex];
		Triangle inverse{
			{
				triangle.vertices[2],
				triangle.vertices[1],
				triangle.vertices[0]
			},
			// FIXME UV coordinates
			{},
			triangle.normals
		};
		
		for (uint32_t normalIndex = 0; normalIndex < triangle.normals.size(); normalIndex++)
		{
			normalBuffer[triangle.normals[normalIndex]] *= -1.0f;
		}
		
		triangle = inverse;
	}
}

Mesh Mesh::cube(const float sideLength, const uint32_t materialOffset,
				std::vector<Triangle> &triangleBuffer, std::vector<Vertex> &vertexBuffer, std::vector<Math::Vector4> &normalBuffer)
{
	Mesh returnValue;
	
	// Each vertex is offset by half the side length on two axes
	float halfSideLength = sideLength / 2.0f;
	
	// Create vertices, a cube has four of them
	Math::Vector4 v0, v1, v2, v3, v4, v5, v6, v7;
	
	// Upper four
	v0 = {-halfSideLength, halfSideLength, halfSideLength};
	v1 = {halfSideLength, halfSideLength, halfSideLength};
	v2 = {halfSideLength, halfSideLength, -halfSideLength};
	v3 = {-halfSideLength, halfSideLength, -halfSideLength};
	
	// Lower four
	v4 = {-halfSideLength, -halfSideLength, halfSideLength};
	v5 = {halfSideLength, -halfSideLength, halfSideLength};
	v6 = {halfSideLength, -halfSideLength, -halfSideLength};
	v7 = {-halfSideLength, -halfSideLength, -halfSideLength};
	
	returnValue.materialOffset = materialOffset;
	returnValue.triangleOffset = uint32_t(triangleBuffer.size());
	returnValue.triangleCount = 12;
	returnValue.vertexOffset = uint32_t(vertexBuffer.size());
	returnValue.vertexCount = 8;
	returnValue.normalOffset = uint32_t(normalBuffer.size());
	returnValue.normalCount = 6;
	
	std::vector<Vertex> vertices{v0, v1, v2, v3, v4, v5, v6, v7};
	
	// Create Triangles
	std::vector<Triangle> triangles = {
		// Upper face
		{{0, 1, 2}},
		{{0, 2, 3}},
		// Lower face
		{{6, 5, 4}},
		{{7, 6, 4}},
		// Front face
		{{4, 5, 1}},
		{{4, 1, 0}},
		// Back face
		{{6, 7, 3}},
		{{6, 3, 2}},
		// Left face
		{{4, 0, 3}},
		{{4, 3, 7}},
		// Right face
		{{5, 6, 2}},
		{{5, 2, 1}},
	};
	
	for (Triangle &triangle : triangles)
	{	
		for (uint32_t &vertexIndex : triangle.vertices)
		{
			vertexIndex += vertexBuffer.size();
		}
	}
	
	vertexBuffer.insert(vertexBuffer.end(), vertices.begin(), vertices.end());
	triangleBuffer.insert(triangleBuffer.end(), triangles.begin(), triangles.end());
	
	for (uint32_t triangleIndex = 0; triangleIndex < triangles.size(); triangleIndex++)
	{
		Triangle *triangle = &triangleBuffer[returnValue.triangleOffset + triangleIndex];
		const Math::Vector4 normal = Triangle::normal(triangle, vertexBuffer.data());
		const uint32_t normalIndex = uint32_t(normalBuffer.size());
		
		triangle->normals = {normalIndex, normalIndex, normalIndex};
		normalBuffer.push_back(normal);
	}
	
	return returnValue;
}

Mesh Mesh::plane(const float sideLength, const uint32_t materialOffset,
				 std::vector<Triangle> &triangleBuffer, std::vector<Vertex> &vertexBuffer, std::vector<Math::Vector4> &normalBuffer)
{
	Mesh returnValue(materialOffset);
	
	float halfSideLength = sideLength / 2.0f;
	Math::Vector4 v0, v1, v2, v3;
	Math::Vector4 n;
	
	v0 = {-halfSideLength, 0, halfSideLength};
	v1 = {halfSideLength, 0, halfSideLength};
	v2 = {halfSideLength, 0, -halfSideLength};
	v3 = {-halfSideLength, 0, -halfSideLength};
	
	// FIXME finish
	
//	n = Triangle{{v0, v1, v2}}.normal();
	
//	returnValue._triangles = {
//		{{v0, v1, v2}, {n, n, n}},
//		{{v0, v2, v3}, {n, n, n}}
//	};
	
	return returnValue;
}

Mesh Mesh::sphere(const float radius, const uint32_t horizontalSubDivisions, const uint32_t verticalSubDivisions, const uint32_t materialOffset,
				  std::vector<Triangle> &triangleBuffer, std::vector<Vertex> &vertexBuffer, std::vector<Math::Vector4> &normalBuffer)
{
	Mesh returnValue(materialOffset);
	
	std::vector<Vertex> vertices;
	std::vector<Math::Vector4> normals;
	std::vector<Triangle> triangles;
	
	// Generate vertices
	for (uint32_t vertical = 0; vertical <= verticalSubDivisions; vertical++)
	{
		const float theta = float(M_PI) * (float(vertical) / float(verticalSubDivisions));
		
		for (uint32_t horizontal = 0; horizontal <= horizontalSubDivisions; horizontal++)
		{
			const float phi = float(M_PI) * 2.0f * (float(horizontal) / float(horizontalSubDivisions));
			
			const Math::Vector4 v = sphericalToCartesian(phi, theta, radius);
			
			vertices.push_back(v);
			
			normals.push_back(v.normalized());
		}
	}
	
	// Generate triangles
	for (uint32_t vertical = 0; vertical < verticalSubDivisions; vertical++)
	{
		for (uint32_t horizontal = 0; horizontal < horizontalSubDivisions; horizontal++)
		{
			uint32_t v0, v1, v2, v3;
			
			v0 = (horizontalSubDivisions + 1) * vertical + horizontal;
			v1 = (horizontalSubDivisions + 1) * vertical + horizontal + 1;
			v2 = (horizontalSubDivisions + 1) * (vertical + 1) + horizontal + 1;
			v3 = (horizontalSubDivisions + 1) * (vertical + 1) + horizontal;
			
			if (vertical != verticalSubDivisions - 1)
			{
				triangles.push_back({{v0, v2, v3}, {}, {v0, v2, v3}});
			}
			
			if (vertical != 0)
			{
				triangles.push_back({{v0, v1, v2}, {}, {v0, v1, v2}});
			}
		}
	}
	
	returnValue.materialOffset = materialOffset;
	returnValue.triangleOffset = uint32_t(triangleBuffer.size());
	returnValue.triangleCount = uint32_t(triangles.size());
	returnValue.vertexOffset = uint32_t(vertexBuffer.size());
	returnValue.vertexCount = uint32_t(vertices.size());
	returnValue.normalOffset = uint32_t(normalBuffer.size());
	returnValue.normalCount = uint32_t(normals.size());
	
	for (uint32_t triangleIndex = 0; triangleIndex < returnValue.triangleCount; triangleIndex++)
	{
		for (uint32_t &vertexIndex : triangles[triangleIndex].vertices)
		{
			vertexIndex += returnValue.vertexOffset;
		}
		
		for (uint32_t &normalIndex : triangles[triangleIndex].normals)
		{
			normalIndex += returnValue.normalOffset;
		}
	}
	
	vertexBuffer.insert(vertexBuffer.end(), vertices.cbegin(), vertices.cend());
	normalBuffer.insert(normalBuffer.end(), normals.cbegin(), normals.cend());
	triangleBuffer.insert(triangleBuffer.end(), triangles.cbegin(), triangles.cend());
	
	return returnValue;
}

Math::Vector4 Mesh::sphericalToCartesian(const float horizontal, const float vertical, const float radius)
{
	Math::Vector4 returnValue;
	
	const float x = radius * std::sin(vertical) * std::cos(horizontal);
	const float y = radius * std::cos(vertical);
	const float z = radius * std::sin(vertical) * std::sin(horizontal);
	
	returnValue = {x, y, z};
	
	return returnValue;
}

}
