#ifndef KDTREEBUILDER_H
#define KDTREEBUILDER_H

#include <memory>
#include <vector>

#include "geometrycontainer.h"
#include "math/vector4.h"
#include "mesh.h"
#include "triangle.h"

#include <cxxutility/definitions.h>

#ifndef CXX_NVCC
#include <cxxutility/debugstream.h>
#endif

namespace ToyPT::Rendering
{

enum Axis
{
	X	= 0,
	Y	= 1,
	Z	= 2
};

struct Box
{
	Math::Vector4 min;
	Math::Vector4 max;
};

struct Leaf
{
	uint32_t triangleIndex	= 0;
	uint32_t meshIndex		= 0;
};

struct Node
{
	Axis					axis		= Axis::X;
	Box						boundingBox;
	std::unique_ptr<Node>	left;
	std::unique_ptr<Node>	right;
	std::vector<Leaf>		leafs;
};

#ifndef CXX_NVCC
inline std::ostream &operator<<(std::ostream &stream, const Axis &object)
{
	switch (object)
	{
		case Axis::X:
			stream << "Axis::X";
			break;
		case Axis::Y:
			stream << "Axis::X";
			break;
		case Axis::Z:
			stream << "Axis::X";
			break;
	}
	
	return stream;
}

inline std::ostream &operator<<(std::ostream &stream, const Box &object)
{
	stream << "Box{min=" << object.min << ", max=" << object.max << "}";
	
	return stream;
}

inline std::ostream &operator<<(std::ostream &stream, const Leaf &object)
{
	stream << "Leaf{triangleIndex=" << object.triangleIndex << ", meshIndex=" << object.meshIndex << "}";
	
	return stream;
}

inline std::ostream &operator<<(std::ostream &stream, const Node &object)
{
	stream << "";
	
	return stream;
}
#endif

class KdTreeBuilder
{
public:
	void			build(const Obj::GeometryContainer &geometry);
	
	Node			*root();
	
	static uint32_t	treeSize(const Node *node);
	
private:
	static constexpr uint32_t	_threshold = 128;
	std::unique_ptr<Node>		_root;
	
	Axis	_nextAxis(
				Axis axis);
	
	Box		_boundingBox(
				const Obj::GeometryContainer &geometry,
				const Node &node) const;
	
	Box		_boundingBox(
				const Obj::GeometryContainer &geometry,
				const Obj::Triangle &triangle) const;
	
	void	_initializeRoot(
				const Obj::GeometryContainer &geometry);
	
	void	_split(
				const Obj::GeometryContainer &geometry,
				Node &node);
};

}

#endif // KDTREEBUILDER_H
