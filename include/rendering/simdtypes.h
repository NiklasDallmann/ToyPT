#ifndef SIMDTYPES_H
#define SIMDTYPES_H

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include <vector4.h>

namespace Rendering::Simd
{

struct CoordinateBuffer
{
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	
	uint32_t size() const;
	void append(const Math::Vector4 &vector);
};

struct Mesh
{
	uint32_t vertexOffset;
	uint32_t vertexCount;
	uint32_t materialOffset;
};

static constexpr uint32_t triangleVertexCount = 3;

using VertexBuffer = CoordinateBuffer;
using NormalBuffer = CoordinateBuffer;
using MeshBuffer = std::vector<Mesh>;

} // namespace Rendering::Simd

#endif // SIMDTYPES_H
