#ifndef CUDATYPES_H
#define CUDATYPES_H

#include <stdint.h>
#include <material.h>
#include <math/vector4.h>

namespace Rendering
{
namespace Cuda
{
namespace Types
{

struct Triangle
{
	Math::Vector4 v0;
	Math::Vector4 e01;
	Math::Vector4 e02;
	Math::Vector4 e12;
	Math::Vector4 n0;
	Math::Vector4 n1;
	Math::Vector4 n2;
	uint32_t meshIndex;
};

struct Mesh
{
	uint32_t triangleOffset;
	uint32_t triangleCount;
	uint32_t materialOffset;
};

struct IntersectionInfo
{
	const Mesh *mesh = nullptr;
	uint32_t triangleOffset = 0xFFFFFFFF;
	float u = 0.0f;
	float v = 0.0f;
};

struct Scene
{
	const Triangle *triangleBuffer = nullptr;
	const uint32_t triangleCount = 0u;
	const Mesh *meshBuffer = nullptr;
	const uint32_t meshCount = 0u;
	const Material *materialBuffer = nullptr;
	const uint32_t materialCount = 0u;
};

struct Tile
{
	uint32_t x0;
	uint32_t y0;
	uint32_t x1;
	uint32_t y1;
};

} // namespace Types
} // namespace Cuda
} // namespace Rendering

#endif // CUDATYPES_H
