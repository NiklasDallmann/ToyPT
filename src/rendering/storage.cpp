#include "obj/geometrycontainer.h"
#include "obj/mesh.h"
#include "storage.h"

#include <debugstream.h>

namespace ToyPT::Rendering::Storage
{

uint32_t CoordinateBuffer::size() const
{
	return uint32_t(this->x.size());
}

void CoordinateBuffer::append(const Math::Vector4 &vector)
{
	this->x.push_back(vector.x());
	this->y.push_back(vector.y());
	this->z.push_back(vector.z());
}

uint32_t PreComputedTriangleBuffer::size() const
{
	return this->v0.size();
}

void PreComputedTriangleBuffer::append(const Math::Vector4 &v0, const Math::Vector4 &v1, const Math::Vector4 &v2,
									   const Math::Vector4 &n0, const Math::Vector4 &n1, const Math::Vector4 &n2,
									   const Mask mask, const MeshOffset mesh)
{
	this->v0.append(v0);
	this->v1.append(v1);
	this->v2.append(v2);
	this->e01.append(v1 - v0);
	this->e02.append(v2 - v0);
	this->n0.append(n0);
	this->n1.append(n1);
	this->n2.append(n2);
	this->mask.push_back(mask);
	this->mesh.push_back(mesh);
}

PrecomputedTrianglePointer PreComputedTriangleBuffer::begin()
{
	return this->data();
}

PrecomputedTrianglePointer PreComputedTriangleBuffer::end()
{
	return this->begin() + this->size();
}

PrecomputedTriangle PreComputedTriangleBuffer::operator[](const uint32_t index) const
{
	PrecomputedTriangle returnValue;
	
	returnValue.v0		= {this->v0.x[index], this->v0.y[index], this->v0.z[index]};
	returnValue.v1		= {this->v1.x[index], this->v1.y[index], this->v1.z[index]};
	returnValue.v2		= {this->v2.x[index], this->v2.y[index], this->v2.z[index]};
	returnValue.e01		= {this->e01.x[index], this->e01.y[index], this->e01.z[index]};
	returnValue.e02		= {this->e02.x[index], this->e02.y[index], this->e02.z[index]};
	returnValue.n0		= {this->n0.x[index], this->n0.y[index], this->n0.z[index]};
	returnValue.n1		= {this->n1.x[index], this->n1.y[index], this->n1.z[index]};
	returnValue.n2		= {this->n2.x[index], this->n2.y[index], this->n2.z[index]};
	returnValue.mask	= this->mask[index];
	returnValue.mesh	= this->mesh[index];
	
	return returnValue;
}

void geometryToBuffer(const Obj::GeometryContainer &geometry, PreComputedTriangleBuffer &triangleBuffer, MeshBuffer &meshBuffer)
{
	for (uint32_t meshIndex = 0; meshIndex < geometry.meshBuffer.size(); meshIndex++)
	{
		const Obj::Mesh &objMesh = geometry.meshBuffer[meshIndex];
		Storage::Mesh mesh;
		mesh.triangleOffset = triangleBuffer.size();
		mesh.triangleCount = objMesh.triangleCount;
		mesh.materialOffset = objMesh.materialOffset;
		
		for (uint32_t triangleIndex = 0; triangleIndex < objMesh.triangleCount; triangleIndex++)
		{
			const Obj::Triangle &triangle = geometry.triangleBuffer[objMesh.triangleOffset + triangleIndex];
			
			Storage::Vertex v0, v1, v2;
			Storage::Normal n0, n1, n2;
			
			v0 = geometry.vertexBuffer[triangle.vertices[0]];
			v1 = geometry.vertexBuffer[triangle.vertices[1]];
			v2 = geometry.vertexBuffer[triangle.vertices[2]];
			
			n0 = geometry.normalBuffer[triangle.normals[0]];
			n1 = geometry.normalBuffer[triangle.normals[1]];
			n2 = geometry.normalBuffer[triangle.normals[2]];
			
			triangleBuffer.append(v0, v1, v2, n0, n1, n2, Storage::maskTrue, meshIndex);
		}
		
		meshBuffer.push_back(mesh);
	}
}

}
