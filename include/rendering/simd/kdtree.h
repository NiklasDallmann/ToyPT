#ifndef KDTREE_H
#define KDTREE_H

#include <math/simd/vector3pack.h>
#include <math/vector4.h>
#include <memory>
#include "storage.h"

#include "ray.h"

namespace ToyPT::Rendering::Simd
{

class KdTree
{
public:
	using NodeType = KdTree;
	using LeafType = Storage::PreComputedTriangleBuffer;
	
	KdTree(NodeType *left, NodeType *right) :
		_left(left),
		_right(right)
	{
	}
	
	KdTree(LeafType *leaf = nullptr) :
		_leaf(leaf)
	{
	}
	
	bool hasLeaf() const
	{
		return (this->_leaf.get() != nullptr);
	}
	
	LeafType *traverse(const Ray &ray);
	
	KdTree fromBuffer(const Storage::PreComputedTriangleBuffer &buffer);
	
private:
	enum class Axis
	{
		X,
		Y,
		Z
	};
	
	Math::Vector4 _min;
	Math::Vector4 _max;
	
	std::unique_ptr<NodeType> _left;
	std::unique_ptr<NodeType> _right;
	std::unique_ptr<LeafType> _leaf;
	
	inline bool _intersect(const Ray &ray);
	float _minX(const Storage::PrecomputedTriangle &triangle);
	float _minY(const Storage::PrecomputedTriangle &triangle);
	float _minZ(const Storage::PrecomputedTriangle &triangle);
	float _maxX(const Storage::PrecomputedTriangle &triangle);
	float _maxY(const Storage::PrecomputedTriangle &triangle);
	float _maxZ(const Storage::PrecomputedTriangle &triangle);
	Math::Vector4 _minPoint(const Storage::PreComputedTriangleBuffer &buffer);
	Math::Vector4 _maxPoint(const Storage::PreComputedTriangleBuffer &buffer);
};

} // namespace ToyPT::Rendering::Simd

#endif // KDTREE_H
