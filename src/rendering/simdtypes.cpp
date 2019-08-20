#include "simdtypes.h"

namespace Rendering::Simd
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

}
