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
	
	CoordinateBufferPointer &operator++(int);
	CoordinateBufferPointer &operator+=(const uint32_t offset);
};

CoordinateBufferPointer operator+(const CoordinateBufferPointer &pointer, const uint32_t offset);

struct CoordinateBuffer
{
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	
	uint32_t size() const;
	void append(const Math::Vector4 &vector);
	CoordinateBufferPointer data();
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
	
	PrecomputedTrianglePointer &operator++(int);
	PrecomputedTrianglePointer &operator+=(const uint32_t offset);
};

PrecomputedTrianglePointer operator+(const PrecomputedTrianglePointer &pointer, const uint32_t offset);

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
	PrecomputedTrianglePointer data();
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
