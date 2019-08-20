#include <cmath>
#include <math.h>

#include "geometrycontainer.h"
#include "mesh.h"

namespace Rendering::Obj
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

void Mesh::transform(const Math::Matrix4x4 &matrix, GeometryContainer &container)
{
	for (uint32_t vertexIndex = this->vertexOffset; vertexIndex < (this->vertexOffset + vertexCount); vertexIndex++)
	{
		container.vertexBuffer[vertexIndex] = matrix * container.vertexBuffer[vertexIndex];
	}
	
	for (uint32_t normalIndex = this->normalOffset; normalIndex < (this->normalOffset + normalCount); normalIndex++)
	{
		// FIXME normalize
		container.normalBuffer[normalIndex] = matrix * container.normalBuffer[normalIndex];
	}
}

void Mesh::translate(const Math::Vector4 &vector, GeometryContainer &container)
{
	for (uint32_t vertexIndex = this->vertexOffset; vertexIndex < (this->vertexOffset + this->vertexCount); vertexIndex++)
	{
		container.vertexBuffer[vertexIndex] += vector;
	}
}

void Mesh::invert(GeometryContainer &container)
{
	for (uint32_t triangleIndex = this->triangleOffset; triangleIndex < (this->triangleOffset + this->triangleCount); triangleIndex++)
	{
		Triangle &triangle = container.triangleBuffer[triangleIndex];
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
			container.normalBuffer[triangle.normals[normalIndex]] *= -1.0f;
		}
		
		triangle = inverse;
	}
}

Mesh Mesh::cube(const float sideLength, const uint32_t materialOffset,
				GeometryContainer &container)
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
	returnValue.triangleOffset = uint32_t(container.triangleBuffer.size());
	returnValue.triangleCount = 12;
	returnValue.vertexOffset = uint32_t(container.vertexBuffer.size());
	returnValue.vertexCount = 8;
	returnValue.normalOffset = uint32_t(container.normalBuffer.size());
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
			vertexIndex += container.vertexBuffer.size();
		}
	}
	
	container.vertexBuffer.insert(container.vertexBuffer.end(), vertices.begin(), vertices.end());
	container.triangleBuffer.insert(container.triangleBuffer.end(), triangles.begin(), triangles.end());
	
	for (uint32_t triangleIndex = 0; triangleIndex < triangles.size(); triangleIndex++)
	{
		Triangle *triangle = &container.triangleBuffer[returnValue.triangleOffset + triangleIndex];
		const Math::Vector4 normal = Triangle::normal(triangle, container.vertexBuffer.data());
		const uint32_t normalIndex = uint32_t(container.normalBuffer.size());
		
		triangle->normals = {normalIndex, normalIndex, normalIndex};
		container.normalBuffer.push_back(normal);
	}
	
	return returnValue;
}

Mesh Mesh::plane(const float sideLength, const uint32_t materialOffset,
				 GeometryContainer &container)
{
	Mesh returnValue(materialOffset);
	
	float halfSideLength = sideLength / 2.0f;
	Math::Vector4 v0, v1, v2, v3;
	
	v0 = {-halfSideLength, 0, halfSideLength};
	v1 = {halfSideLength, 0, halfSideLength};
	v2 = {halfSideLength, 0, -halfSideLength};
	v3 = {-halfSideLength, 0, -halfSideLength};
	
	std::vector<Vertex> vertices{v0, v1, v2, v3};
	std::vector<Triangle> triangles{
		{{0, 1, 2}, {}, {0, 0, 0}},
		{{0, 2, 3}, {}, {0, 0, 0}}
	};
	std::vector<Math::Vector4> normals{Triangle::normal(triangles.data(), vertices.data())};
	
	returnValue.materialOffset = materialOffset;
	returnValue.triangleOffset = uint32_t(container.triangleBuffer.size());
	returnValue.triangleCount = uint32_t(triangles.size());
	returnValue.vertexOffset = uint32_t(container.vertexBuffer.size());
	returnValue.vertexCount = uint32_t(vertices.size());
	returnValue.normalOffset = uint32_t(container.normalBuffer.size());
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
	
	container.vertexBuffer.insert(container.vertexBuffer.end(), vertices.cbegin(), vertices.cend());
	container.normalBuffer.insert(container.normalBuffer.end(), normals.cbegin(), normals.cend());
	container.triangleBuffer.insert(container.triangleBuffer.end(), triangles.cbegin(), triangles.cend());
	
	return returnValue;
}

Mesh Mesh::sphere(const float radius, const uint32_t horizontalSubDivisions, const uint32_t verticalSubDivisions, const uint32_t materialOffset,
				  GeometryContainer &container)
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
	returnValue.triangleOffset = uint32_t(container.triangleBuffer.size());
	returnValue.triangleCount = uint32_t(triangles.size());
	returnValue.vertexOffset = uint32_t(container.vertexBuffer.size());
	returnValue.vertexCount = uint32_t(vertices.size());
	returnValue.normalOffset = uint32_t(container.normalBuffer.size());
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
	
	container.vertexBuffer.insert(container.vertexBuffer.end(), vertices.cbegin(), vertices.cend());
	container.normalBuffer.insert(container.normalBuffer.end(), normals.cbegin(), normals.cend());
	container.triangleBuffer.insert(container.triangleBuffer.end(), triangles.cbegin(), triangles.cend());
	
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
