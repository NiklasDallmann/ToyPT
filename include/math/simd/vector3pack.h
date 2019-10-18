#ifndef Vector3Pack_H
#define Vector3Pack_H

#include <immintrin.h>
#include <storage.h>

namespace ToyPT
{
namespace Math
{
namespace Simd
{

class Vector3Pack
{
public:
	__m256 x;
	__m256 y;
	__m256 z;
	
	Vector3Pack()
	{
	}
	
	Vector3Pack(const __m256 &vector)
	{
		this->x = vector;
		this->y = vector;
		this->z = vector;
	}
	
	void loadUnaligned(const float *x, const float *y, const float *z)
	{
		this->x = _mm256_loadu_ps(x);
		this->y = _mm256_loadu_ps(y);
		this->z = _mm256_loadu_ps(z);
	}
	
	void loadUnaligned(Rendering::Storage::CoordinateBufferPointer data)
	{
		this->loadUnaligned(data.x, data.y, data.z);
	}
	
	__m256 dotProduct(const Vector3Pack &other) const
	{
		__m256 returnValue;
		
		returnValue = _mm256_fmadd_ps(this->x, other.x, _mm256_fmadd_ps(this->y, other.y, _mm256_mul_ps(this->z, other.z)));
		
		return returnValue;
	}
	
	Vector3Pack crossProduct(const Vector3Pack &other) const
	{
		Vector3Pack returnValue;
		
		returnValue.x = _mm256_fmsub_ps(this->y, other.z, _mm256_mul_ps(this->z, other.y));
		returnValue.y = _mm256_fmsub_ps(this->z, other.x, _mm256_mul_ps(this->x, other.z));
		returnValue.z = _mm256_fmsub_ps(this->x, other.y, _mm256_mul_ps(this->y, other.x));
		
		return returnValue;
	}
	
	static Vector3Pack blend(const Vector3Pack &a, const Vector3Pack &b, const __m256 &mask)
	{
		Vector3Pack returnValue;
		
		returnValue.x = _mm256_blendv_ps(a.x, b.x, mask);
		returnValue.y = _mm256_blendv_ps(a.y, b.y, mask);
		returnValue.z = _mm256_blendv_ps(a.z, b.z, mask);
		
		return returnValue;
	}
	
	Vector3Pack &operator+=(const Vector3Pack &other)
	{
		this->x = _mm256_add_ps(this->x, other.x);
		this->y = _mm256_add_ps(this->y, other.y);
		this->z = _mm256_add_ps(this->z, other.z);
		
		return *this;
	}
	
	Vector3Pack &operator-=(const Vector3Pack &other)
	{
		this->x = _mm256_sub_ps(this->x, other.x);
		this->y = _mm256_sub_ps(this->y, other.y);
		this->z = _mm256_sub_ps(this->z, other.z);
		
		return *this;
	}
	
	Vector3Pack &operator*=(const Vector3Pack &other)
	{
		this->x = _mm256_mul_ps(this->x, other.x);
		this->y = _mm256_mul_ps(this->y, other.y);
		this->z = _mm256_mul_ps(this->z, other.z);
		
		return *this;
	}
	
	Vector3Pack &operator/=(const Vector3Pack &other)
	{
		this->x = _mm256_div_ps(this->x, other.x);
		this->y = _mm256_div_ps(this->y, other.y);
		this->z = _mm256_div_ps(this->z, other.z);
		
		return *this;
	}
};

inline Vector3Pack operator+(const Vector3Pack &left, const Vector3Pack &right)
{
	Vector3Pack returnValue = left;
	
	returnValue += right;
	
	return returnValue;
}

inline Vector3Pack operator-(const Vector3Pack &left, const Vector3Pack &right)
{
	Vector3Pack returnValue = left;
	
	returnValue -= right;
	
	return returnValue;
}

inline Vector3Pack operator*(const Vector3Pack &left, const Vector3Pack &right)
{
	Vector3Pack returnValue = left;
	
	returnValue *= right;
	
	return returnValue;
}

inline Vector3Pack operator/(const Vector3Pack &left, const Vector3Pack &right)
{
	Vector3Pack returnValue = left;
	
	returnValue /= right;
	
	return returnValue;
}

} // namespace Simd
} // namespace Math
} // namespace ToyPT

#endif // Vector3Pack_H
