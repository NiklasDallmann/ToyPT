#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <algorithm>
#include <immintrin.h>

namespace Math
{

constexpr float epsilon = 1e-6;

template <typename T>
inline bool fuzzyCompareEqual(const T left, const T right);

template<>
inline bool fuzzyCompareEqual<float>(const float left, const float right)
{
	bool returnValue = false;
	
	returnValue = std::abs(std::abs(left) - std::abs(right)) < epsilon;
	
	return returnValue;
}

template<typename T>
inline T lerp(const T a, const T b, const float t)
{
	return (1.0f - t) * b + t * a;
}

template<>
inline __m256 lerp<__m256>(const __m256 a, const __m256 b, const float t)
{
	__m256 tVector = _mm256_set1_ps(t);
	__m256 inverseTVector = _mm256_set1_ps(1.0f - t);
	
	return _mm256_add_ps(_mm256_mul_ps(inverseTVector, b), _mm256_mul_ps(tVector, a));
}


} // namespace Math

#endif // ALGORITHMS_H
