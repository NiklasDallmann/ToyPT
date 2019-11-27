#include <algorithm>
#include <limits>
#include <stack>
#include <vector>

#include "kdtreebuilder.h"

namespace ToyPT::Rendering
{

void KdTreeBuilder::build(const Obj::GeometryContainer &geometry)
{
	std::stack<Node *, std::vector<Node *>> stack;
	
	_initializeRoot(geometry);
	
	stack.push(this->_root.get());
	
	while (!stack.empty())
	{
		Node *parent = stack.top();
		
		_split(geometry, *parent);
		stack.pop();
		
		if (parent->right->leafs.size() > KdTreeBuilder::_threshold)
		{
			stack.push(parent->right.get());
		}
		
		if (parent->left->leafs.size() > KdTreeBuilder::_threshold)
		{
			stack.push(parent->left.get());
		}
	}
}

Node *KdTreeBuilder::root()
{
	return this->_root.get();
}

Axis KdTreeBuilder::_nextAxis(ToyPT::Rendering::Axis axis)
{
	Axis returnValue;
	
	switch (axis)
	{
		case Axis::X:
			returnValue = Axis::Y;
			break;
		case Axis::Y:
			returnValue = Axis::Z;
			break;
		case Axis::Z:
			returnValue = Axis::X;
			break;
	}
	
	return returnValue;
}

Box KdTreeBuilder::_boundingBox(const Obj::GeometryContainer &geometry, const Node &node) const
{
	Box returnValue{{std::numeric_limits<float>::max()}, {std::numeric_limits<float>::lowest()}};
	
	for (const Leaf &leaf : node.leafs)
	{
		const Obj::Triangle	&triangle			= geometry.triangleBuffer[leaf.triangleIndex];
		const Box			triangleBoundingBox	= _boundingBox(geometry, triangle);
		
		for (uint32_t i = 0; i < 3; i++)
		{
			if (triangleBoundingBox.min[i] < returnValue.min[i])
			{
				returnValue.min[i] = triangleBoundingBox.min[i];
			}
			
			if (triangleBoundingBox.max[i] > returnValue.max[i])
			{
				returnValue.max[i] = triangleBoundingBox.max[i];
			}
		}
	}
	
	return returnValue;
}

Box KdTreeBuilder::_boundingBox(const Obj::GeometryContainer &geometry, const Obj::Triangle &triangle) const
{
	Box returnValue{{std::numeric_limits<float>::max()}, {std::numeric_limits<float>::lowest()}};
	
	for (const uint32_t &vertexIndex : triangle.vertices)
	{
		const Math::Vector4 &vertex = geometry.vertexBuffer[vertexIndex];
		
		for (uint32_t i = 0; i < 3; i++)
		{
			if (vertex[i] < returnValue.min[i])
			{
				returnValue.min[i] = vertex[i];
			}
			
			if (vertex[i] > returnValue.max[i])
			{
				returnValue.max[i] = vertex[i];
			}
		}
	}
	
	return returnValue;
}

void KdTreeBuilder::_initializeRoot(const Obj::GeometryContainer &geometry)
{
	this->_root = std::make_unique<Node>();
	
	for (uint32_t meshIndex = 0; meshIndex < geometry.meshBuffer.size(); meshIndex++)
	{
		const Obj::Mesh &objMesh = geometry.meshBuffer[meshIndex];
		
		for (uint32_t triangleIndex = 0; triangleIndex < objMesh.triangleCount; triangleIndex++)
		{
			this->_root->leafs.push_back(Leaf{triangleIndex, meshIndex});
		}
	}
	
	this->_root->boundingBox = this->_boundingBox(geometry, *this->_root);
}

void KdTreeBuilder::_split(const Obj::GeometryContainer &geometry, Node &node)
{	
	std::sort(node.leafs.begin(), node.leafs.end(), [this, &geometry, &node](Leaf &left, Leaf &right)
	{
		return	_boundingBox(geometry, geometry.triangleBuffer[left.triangleIndex]).min[node.axis] < 
				_boundingBox(geometry, geometry.triangleBuffer[right.triangleIndex]).min[node.axis];
	});
	
	node.left				= std::make_unique<Node>();
	node.right				= std::make_unique<Node>();
	
	node.left->axis			= _nextAxis(node.axis);
	node.right->axis		= _nextAxis(node.axis);
	
	const float splitPoint	= _boundingBox(geometry, geometry.triangleBuffer[node.leafs[node.leafs.size() / 2].triangleIndex]).min[node.axis];
	
	for (uint32_t i = 0; i < uint32_t(node.leafs.size()) / 2; i++)
	{
		node.left->leafs.push_back((node.leafs[i]));
		
		if (_boundingBox(geometry, geometry.triangleBuffer[node.leafs[i].triangleIndex]).max[node.axis] >= splitPoint)
		{
			node.right->leafs.push_back(((node.leafs[i])));
		}
	}
	
	for (uint32_t i = uint32_t(node.leafs.size()) / 2; i < node.leafs.size(); i++)
	{
		node.right->leafs.push_back(node.leafs[i]);
	}
	
	node.leafs.clear();
}

}
