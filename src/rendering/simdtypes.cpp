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

CoordinateBufferPointer CoordinateBuffer::data()
{
	return {this->x.data(), this->y.data(), this->z.data()};
}

CoordinateBufferPointer &CoordinateBufferPointer::operator++(int)
{
	this->x++;
	this->y++;
	this->z++;
	
	return *this;
}

CoordinateBufferPointer &CoordinateBufferPointer::operator+=(const uint32_t offset)
{
	this->x += offset;
	this->y += offset;
	this->z += offset;
	
	return *this;
}

uint32_t PreComputedTriangleBuffer::size() const
{
	return this->v0.size();
}

void PreComputedTriangleBuffer::append(const Math::Vector4 &v0, const Math::Vector4 &v1, const Math::Vector4 &v2, const Math::Vector4 &n0, const Math::Vector4 &n1, const Math::Vector4 &n2, const uint32_t m)
{
	this->v0.append(v0);
	this->v1.append(v1);
	this->v2.append(v2);
	this->e01.append(v1 - v0);
	this->e02.append(v2 - v0);
	this->n0.append(n0);
	this->n1.append(n1);
	this->n2.append(n2);
	this->m.push_back(m);
}

PrecomputedTrianglePointer PreComputedTriangleBuffer::data()
{
	return {this->v0.data(), this->v1.data(), this->v2.data(),
			this->e01.data(), this->e02.data(),
			this->n0.data(), this->n1.data(), this->n2.data(),
			this->m.data()};
}

PrecomputedTriangle PreComputedTriangleBuffer::operator[](const uint32_t index)
{
	PrecomputedTriangle returnValue;
	
	returnValue.v0 = {this->v0.x[index], this->v0.y[index], this->v0.z[index]};
	returnValue.v1 = {this->v1.x[index], this->v1.y[index], this->v1.z[index]};
	returnValue.v2 = {this->v2.x[index], this->v2.y[index], this->v2.z[index]};
	returnValue.e01 = {this->e01.x[index], this->e01.y[index], this->e01.z[index]};
	returnValue.e02 = {this->e02.x[index], this->e02.y[index], this->e02.z[index]};
	returnValue.n0 = {this->n0.x[index], this->n0.y[index], this->n0.z[index]};
	returnValue.n1 = {this->n1.x[index], this->n1.y[index], this->n1.z[index]};
	returnValue.n2 = {this->n2.x[index], this->n2.y[index], this->n2.z[index]};
	returnValue.m = this->m[index];
	
	return returnValue;
}

PrecomputedTrianglePointer &PrecomputedTrianglePointer::operator++(int)
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

PrecomputedTrianglePointer &PrecomputedTrianglePointer::operator+=(const uint32_t offset)
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

CoordinateBufferPointer operator+(const CoordinateBufferPointer &pointer, const uint32_t offset)
{
	CoordinateBufferPointer returnValue = pointer;
	
	returnValue += offset;
	
	return returnValue;
}

PrecomputedTrianglePointer operator+(const PrecomputedTrianglePointer &pointer, const uint32_t offset)
{
	PrecomputedTrianglePointer returnValue = pointer;
	
	returnValue += offset;
	
	return returnValue;
}

}
