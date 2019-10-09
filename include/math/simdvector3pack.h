#ifndef SIMDVECTOR3PACK_H
#define SIMDVECTOR3PACK_H

#include <immintrin.h>

namespace ToyPT
{
namespace Math
{

class SimdVector3Pack
{
public:
	__m256 x;
	__m256 y;
	__m256 z;
	
	SimdVector3Pack()
	{
	}
	
	SimdVector3Pack(const __m256 &vector)
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
	
	__m256 dotProduct(const SimdVector3Pack &other) const
	{
		__m256 returnValue;
		
		returnValue = _mm256_fmadd_ps(this->x, other.x, _mm256_fmadd_ps(this->y, other.y, _mm256_mul_ps(this->z, other.z)));
		
		return returnValue;
	}
	
	SimdVector3Pack crossProduct(const SimdVector3Pack &other) const
	{
		SimdVector3Pack returnValue;
		
		returnValue.x = _mm256_fmsub_ps(this->y, other.z, _mm256_mul_ps(this->z, other.y));
		returnValue.y = _mm256_fmsub_ps(this->z, other.x, _mm256_mul_ps(this->x, other.z));
		returnValue.z = _mm256_fmsub_ps(this->x, other.y, _mm256_mul_ps(this->y, other.x));
		
		return returnValue;
	}
	
	static SimdVector3Pack blend(const SimdVector3Pack &a, const SimdVector3Pack &b, const __m256 &mask)
	{
		SimdVector3Pack returnValue;
		
		returnValue.x = _mm256_blendv_ps(a.x, b.x, mask);
		returnValue.y = _mm256_blendv_ps(a.y, b.y, mask);
		returnValue.z = _mm256_blendv_ps(a.z, b.z, mask);
		
		return returnValue;
	}
	
	SimdVector3Pack &operator+=(const SimdVector3Pack &other)
	{
		this->x = _mm256_add_ps(this->x, other.x);
		this->y = _mm256_add_ps(this->y, other.y);
		this->z = _mm256_add_ps(this->z, other.z);
		
		return *this;
	}
	
	SimdVector3Pack &operator-=(const SimdVector3Pack &other)
	{
		this->x = _mm256_sub_ps(this->x, other.x);
		this->y = _mm256_sub_ps(this->y, other.y);
		this->z = _mm256_sub_ps(this->z, other.z);
		
		return *this;
	}
	
	SimdVector3Pack &operator*=(const SimdVector3Pack &other)
	{
		this->x = _mm256_mul_ps(this->x, other.x);
		this->y = _mm256_mul_ps(this->y, other.y);
		this->z = _mm256_mul_ps(this->z, other.z);
		
		return *this;
	}
	
	SimdVector3Pack &operator/=(const SimdVector3Pack &other)
	{
		this->x = _mm256_div_ps(this->x, other.x);
		this->y = _mm256_div_ps(this->y, other.y);
		this->z = _mm256_div_ps(this->z, other.z);
		
		return *this;
	}
};

inline SimdVector3Pack operator+(const SimdVector3Pack &left, const SimdVector3Pack &right)
{
	SimdVector3Pack returnValue = left;
	
	returnValue += right;
	
	return returnValue;
}

inline SimdVector3Pack operator-(const SimdVector3Pack &left, const SimdVector3Pack &right)
{
	SimdVector3Pack returnValue = left;
	
	returnValue -= right;
	
	return returnValue;
}

inline SimdVector3Pack operator*(const SimdVector3Pack &left, const SimdVector3Pack &right)
{
	SimdVector3Pack returnValue = left;
	
	returnValue *= right;
	
	return returnValue;
}

inline SimdVector3Pack operator/(const SimdVector3Pack &left, const SimdVector3Pack &right)
{
	SimdVector3Pack returnValue = left;
	
	returnValue /= right;
	
	return returnValue;
}

} // namespace Math
} // namespace ToyPT

#endif // SIMDVECTOR3PACK_H
