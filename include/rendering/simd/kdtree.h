#ifndef KDTREE_H
#define KDTREE_H

#include <math/simd/vector3pack.h>
#include <math/vector4.h>
#include <memory>

#include "ray.h"

namespace ToyPT::Rendering::Simd
{

class KdTree
{
public:
	using NodeType = KdTree;
	using LeafType = Math::Simd::Vector3Pack;
	
	KdTree(NodeType *left, NodeType *right) :
		_left(left),
		_right(right)
	{
	}
	
	KdTree(LeafType *leaf) :
		_leaf(leaf)
	{
	}
	
	bool hasLeaf() const
	{
		return (this->_leaf.get() != nullptr);
	}
	
	LeafType *traverse(const Ray &ray);
	
private:
	Math::Vector4 _min;
	Math::Vector4 _max;
	
	std::unique_ptr<NodeType> _left;
	std::unique_ptr<NodeType> _right;
	std::unique_ptr<LeafType> _leaf;
	
	inline bool _intersect(const Ray &ray, float *distance);
};

} // namespace ToyPT::Rendering::Simd

#endif // KDTREE_H
