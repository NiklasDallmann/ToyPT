#include <math/algorithms.h>
#include "simd/kdtree.h"

namespace ToyPT::Rendering::Simd
{

KdTree::LeafType *KdTree::traverse(const Ray &ray)
{
	LeafType *returnValue = nullptr;
	
	if (this->hasLeaf())
	{
		returnValue = this->_leaf.get();
	}
	else
	{
		if (this->_left->_intersect(ray))
		{
			returnValue = this->_left->traverse(ray);
		}
		else if (this->_right->_intersect(ray))
		{
			returnValue = this->_right->traverse(ray);
		}
	}
	
	return returnValue;
}

KdTree KdTree::fromBuffer(const Storage::PreComputedTriangleBuffer &buffer)
{
	KdTree returnValue;
	
	Math::Vector4 min{0};
	Math::Vector4 max{std::numeric_limits<float>::max()};
	
	// Compute box
	for (uint32_t i = 0; i < buffer.size(); i++)
	{
		Storage::PrecomputedTriangle triangle = buffer[i];
		
		
	}
	
	return returnValue;
}

bool KdTree::_intersect(const Ray &ray)
{
	bool returnValue = true;
	
	Math::Vector4 inverseDirection = 1.0f / ray.direction;
	float tmin = (this->_min.x() - ray.origin.x()) * inverseDirection.x();
	float tmax = (this->_max.x() - ray.origin.x()) * inverseDirection.x();
	
	if (tmin > tmax)
	{
		Math::swap(tmin, tmax);
	}
	
	float tymin = (this->_min.y() - ray.origin.y()) * inverseDirection.y();
	float tymax = (this->_max.y() - ray.origin.y()) * inverseDirection.y();
	
	if (tymin > tymax)
	{
		Math::swap(tymin, tymax);
	}
	
	float tzmin = (this->_min.z() - ray.origin.z()) * inverseDirection.z();
	float tzmax = (this->_max.z() - ray.origin.z()) * inverseDirection.z();
	
	if (tzmin > tzmax)
	{
		Math::swap(tzmin, tzmax);
	}
	
	if ((tmin > tymax) | (tymin > tmax))
	{
		returnValue = false;
		return returnValue;
	}
	
	if (tymin > tmin)
	{
		tmin = tymin;
	}
	
	if (tymax < tmax)
	{
		tmax = tymax;
	}
	
	if ((tmin > tzmax) | (tymin > tmax))
	{
		returnValue = false;
		return returnValue;
	}
	
	if (tzmin > tmin)
	{
		tmin = tzmin;
	}
	
	if (tzmax < tmax)
	{
		tmax = tzmax;
	}
	
	return returnValue;
}

float KdTree::_minX(const Storage::PrecomputedTriangle &triangle)
{
	float returnValue = triangle.v0.x();
	
	if (triangle.v1.x() < returnValue)
	{
		returnValue = triangle.v1.x();
	}
	
	if (triangle.v2.x() < returnValue)
	{
		returnValue = triangle.v2.x();
	}
	
	return returnValue;
}

float KdTree::_minY(const Storage::PrecomputedTriangle &triangle)
{
	float returnValue = triangle.v0.y();
	
	if (triangle.v1.y() < returnValue)
	{
		returnValue = triangle.v1.y();
	}
	
	if (triangle.v2.y() < returnValue)
	{
		returnValue = triangle.v2.y();
	}
	
	return returnValue;
}

float KdTree::_minZ(const Storage::PrecomputedTriangle &triangle)
{
	float returnValue = triangle.v0.z();
	
	if (triangle.v1.z() < returnValue)
	{
		returnValue = triangle.v1.z();
	}
	
	if (triangle.v2.z() < returnValue)
	{
		returnValue = triangle.v2.z();
	}
	
	return returnValue;
}

float KdTree::_maxX(const Storage::PrecomputedTriangle &triangle)
{
	float returnValue = triangle.v0.x();
	
	if (triangle.v1.x() > returnValue)
	{
		returnValue = triangle.v1.x();
	}
	
	if (triangle.v2.x() > returnValue)
	{
		returnValue = triangle.v2.x();
	}
	
	return returnValue;
}

float KdTree::_maxY(const Storage::PrecomputedTriangle &triangle)
{
	float returnValue = triangle.v0.y();
	
	if (triangle.v1.y() > returnValue)
	{
		returnValue = triangle.v1.y();
	}
	
	if (triangle.v2.y() > returnValue)
	{
		returnValue = triangle.v2.y();
	}
	
	return returnValue;
}

float KdTree::_maxZ(const Storage::PrecomputedTriangle &triangle)
{
	float returnValue = triangle.v0.z();
	
	if (triangle.v1.z() > returnValue)
	{
		returnValue = triangle.v1.z();
	}
	
	if (triangle.v2.z() > returnValue)
	{
		returnValue = triangle.v2.z();
	}
	
	return returnValue;
}

Math::Vector4 KdTree::_minPoint(const Storage::PreComputedTriangleBuffer &buffer)
{
	Math::Vector4 returnValue;
	
	float minx = std::numeric_limits<float>::max();
	float miny = std::numeric_limits<float>::max();
	float minz = std::numeric_limits<float>::max();
	
	for (uint32_t i = 0; i < buffer.size(); i++)
	{
		const Storage::PrecomputedTriangle triangle = buffer[i];
		
		float x = this->_minX(triangle);
		float y = this->_minY(triangle);
		float z = this->_minZ(triangle);
		
		if (x < minx)
		{
			minx = x;
		}
		
		if (y < miny)
		{
			miny = y;
		}
		
		if (z < minz)
		{
			minz = z;
		}
	}
	
	return returnValue;
}

Math::Vector4 KdTree::_maxPoint(const Storage::PreComputedTriangleBuffer &buffer)
{
	Math::Vector4 returnValue;
	
	float maxx = std::numeric_limits<float>::lowest();
	float maxy = std::numeric_limits<float>::lowest();
	float maxz = std::numeric_limits<float>::lowest();
	
	for (uint32_t i = 0; i < buffer.size(); i++)
	{
		const Storage::PrecomputedTriangle triangle = buffer[i];
		
		float x = this->_maxX(triangle);
		float y = this->_maxY(triangle);
		float z = this->_maxZ(triangle);
		
		if (x < maxx)
		{
			maxx = x;
		}
		
		if (y < maxy)
		{
			maxy = y;
		}
		
		if (z < maxz)
		{
			maxz = z;
		}
	}
	
	return returnValue;
}

}
