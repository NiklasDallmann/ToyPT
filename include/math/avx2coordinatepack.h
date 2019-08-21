#ifndef AVX2COORDINATEPACK_H
#define AVX2COORDINATEPACK_H

#include <immintrin.h>

namespace Math
{

class Avx2CoordinatePack
{
public:
	__m256 x;
	__m256 y;
	__m256 z;
	
	Avx2CoordinatePack()
	{
	}
	
	Avx2CoordinatePack(const __m256 &vector)
	{
		this->x = vector;
		this->y = vector;
		this->z = vector;
	}
	
	__m256 dotProduct(const Avx2CoordinatePack &other) const
	{
		__m256 returnValue;
		
		__m256 x = _mm256_mul_ps(this->x, other.x);
		__m256 y = _mm256_mul_ps(this->y, other.y);
		__m256 z = _mm256_mul_ps(this->z, other.z);
		
		returnValue = _mm256_add_ps(x, _mm256_add_ps(y, z));
		
		return returnValue;
	}
	
	Avx2CoordinatePack crossProduct(const Avx2CoordinatePack &other) const
	{
		Avx2CoordinatePack returnValue;
		
		returnValue.x = _mm256_sub_ps(_mm256_mul_ps(this->y, other.z), _mm256_mul_ps(this->z, other.y));
		returnValue.y = _mm256_sub_ps(_mm256_mul_ps(this->z, other.x), _mm256_mul_ps(this->x, other.z));
		returnValue.z = _mm256_sub_ps(_mm256_mul_ps(this->x, other.y), _mm256_mul_ps(this->y, other.x));
		
		return returnValue;
	}
	
	static Avx2CoordinatePack blend(const Avx2CoordinatePack &a, const Avx2CoordinatePack &b, const __m256 &mask)
	{
		Avx2CoordinatePack returnValue;
		
		returnValue.x = _mm256_blendv_ps(a.x, b.x, mask);
		returnValue.y = _mm256_blendv_ps(a.y, b.y, mask);
		returnValue.z = _mm256_blendv_ps(a.z, b.z, mask);
		
		return returnValue;
	}
	
	Avx2CoordinatePack &operator+=(const Avx2CoordinatePack &other)
	{
		this->x = _mm256_add_ps(this->x, other.x);
		this->y = _mm256_add_ps(this->y, other.y);
		this->z = _mm256_add_ps(this->z, other.z);
		
		return *this;
	}
	
	Avx2CoordinatePack &operator-=(const Avx2CoordinatePack &other)
	{
		this->x = _mm256_sub_ps(this->x, other.x);
		this->y = _mm256_sub_ps(this->y, other.y);
		this->z = _mm256_sub_ps(this->z, other.z);
		
		return *this;
	}
	
	Avx2CoordinatePack &operator*=(const Avx2CoordinatePack &other)
	{
		this->x = _mm256_mul_ps(this->x, other.x);
		this->y = _mm256_mul_ps(this->y, other.y);
		this->z = _mm256_mul_ps(this->z, other.z);
		
		return *this;
	}
	
	Avx2CoordinatePack &operator/=(const Avx2CoordinatePack &other)
	{
		this->x = _mm256_div_ps(this->x, other.x);
		this->y = _mm256_div_ps(this->y, other.y);
		this->z = _mm256_div_ps(this->z, other.z);
		
		return *this;
	}
};

inline Avx2CoordinatePack operator+(const Avx2CoordinatePack &left, const Avx2CoordinatePack &right)
{
	Avx2CoordinatePack returnValue = left;
	
	returnValue += right;
	
	return returnValue;
}

inline Avx2CoordinatePack operator-(const Avx2CoordinatePack &left, const Avx2CoordinatePack &right)
{
	Avx2CoordinatePack returnValue = left;
	
	returnValue -= right;
	
	return returnValue;
}

inline Avx2CoordinatePack operator*(const Avx2CoordinatePack &left, const Avx2CoordinatePack &right)
{
	Avx2CoordinatePack returnValue = left;
	
	returnValue *= right;
	
	return returnValue;
}

inline Avx2CoordinatePack operator/(const Avx2CoordinatePack &left, const Avx2CoordinatePack &right)
{
	Avx2CoordinatePack returnValue = left;
	
	returnValue /= right;
	
	return returnValue;
}

} // namespace Math

#endif // AVX2COORDINATEPACK_H
