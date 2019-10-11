#ifndef STORAGE_H
#define STORAGE_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <vector>

#include <math/vector4.h>

namespace ToyPT::Rendering::Obj{ class GeometryContainer; }

namespace ToyPT::Rendering::Storage
{

struct CoordinateBufferPointer
{
	float *x;
	float *y;
	float *z;
	
	inline CoordinateBufferPointer &operator++(int)
	{
		this->x++;
		this->y++;
		this->z++;
		
		return *this;
	}
	
	inline CoordinateBufferPointer &operator+=(const uint32_t offset)
	{
		this->x += offset;
		this->y += offset;
		this->z += offset;
		
		return *this;
	}
};

inline CoordinateBufferPointer operator+(const CoordinateBufferPointer &pointer, const uint32_t offset)
{
	CoordinateBufferPointer returnValue = pointer;
	
	returnValue += offset;
	
	return returnValue;
}

struct CoordinateBuffer
{
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	
	uint32_t size() const;
	void append(const Math::Vector4 &vector);
	
	inline CoordinateBufferPointer data()
	{
		return {this->x.data(), this->y.data(), this->z.data()};
	}
};

using Vertex = Math::Vector4;
using Normal = Math::Vector4;
using Mask = uint32_t;
using VertexPointer = CoordinateBufferPointer;
using NormalPointer = CoordinateBufferPointer;
using MaskPointer = Mask *;
using VertexBuffer = CoordinateBuffer;
using NormalBuffer = CoordinateBuffer;
using MaskBuffer = std::vector<Mask>;

struct PrecomputedTriangle
{
	Vertex v0;
	Vertex v1;
	Vertex v2;
	Vertex e01;
	Vertex e02;
	Normal n0;
	Normal n1;
	Normal n2;
	Mask m;
};

struct PrecomputedTrianglePointer
{
	VertexPointer v0;
	VertexPointer v1;
	VertexPointer v2;
	VertexPointer e01;
	VertexPointer e02;
	NormalPointer n0;
	NormalPointer n1;
	NormalPointer n2;
	MaskPointer m;
	
	inline PrecomputedTrianglePointer &operator++(int)
	{
		this->v0++;
		this->v1++;
		this->v2++;
		this->e01++;
		this->e02++;
		this->n0++;
		this->n1++;
		this->n2++;
		this->m++;
		
		return *this;
	}
	
	inline PrecomputedTrianglePointer &operator+=(const uint32_t offset)
	{
		this->v0 += offset;
		this->v1 += offset;
		this->v2 += offset;
		this->e01 += offset;
		this->e02 += offset;
		this->n0 += offset;
		this->n1 += offset;
		this->n2 += offset;
		this->m += offset;
		
		return *this;
	}
};

inline PrecomputedTrianglePointer operator+(const PrecomputedTrianglePointer &pointer, const uint32_t offset)
{
	PrecomputedTrianglePointer returnValue = pointer;
	
	returnValue += offset;
	
	return returnValue;
}

struct PreComputedTriangleBuffer
{
	VertexBuffer v0;
	VertexBuffer v1;
	VertexBuffer v2;
	VertexBuffer e01;
	VertexBuffer e02;
	NormalBuffer n0;
	NormalBuffer n1;
	NormalBuffer n2;
	MaskBuffer m;
	
	uint32_t size() const;
	void append(const Math::Vector4 &v0, const Math::Vector4 &v1, const Math::Vector4 &v2,
				const Math::Vector4 &n0, const Math::Vector4 &n1, const Math::Vector4 &n2,
				const uint32_t m);
	
	inline PrecomputedTrianglePointer data()
	{
		return {this->v0.data(), this->v1.data(), this->v2.data(),
				this->e01.data(), this->e02.data(),
				this->n0.data(), this->n1.data(), this->n2.data(),
				this->m.data()};
	}
	
	PrecomputedTriangle operator[](const uint32_t index);
};

struct Mesh
{
	uint32_t triangleOffset;
	uint32_t triangleCount;
	uint32_t materialOffset;
};

using MeshBuffer = std::vector<Mesh>;

void geometryToBuffer(const Obj::GeometryContainer &geometry, Storage::PreComputedTriangleBuffer &triangleBuffer, Storage::MeshBuffer &meshBuffer);

static constexpr uint32_t triangleVertexCount = 3;
static constexpr uint32_t maskTrue = 0xFFFFFFFF;
static constexpr uint32_t maskFalse = 0x00000000;
static constexpr uint32_t avx2FloatCount = sizeof (__m256) / sizeof (float);

} // namespace ToyPT::Rendering::Storage

#endif // STORAGE_H
