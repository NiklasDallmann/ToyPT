#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#ifdef __AVX__
#include <immintrin.h>
#endif

#include <utility/globals.h>
#include "vector4.h"



namespace Math
{

constexpr float epsilon = 1e-6f;

template <typename T>
HOST_DEVICE inline bool fuzzyCompareEqual(const T left, const T right);

template<>
HOST_DEVICE inline bool fuzzyCompareEqual<float>(const float left, const float right)
{
	bool returnValue = false;
	
#ifndef __NVCC__
	returnValue = std::abs(std::abs(left) - std::abs(right)) < epsilon;
#else
	returnValue = fabsf(fabsf(left) - fabsf(right)) < epsilon;
#endif
	
	return returnValue;
}

template<typename T>
HOST_DEVICE inline T lerp(const T a, const T b, const float t)
{
	return (1.0f - t) * b + t * a;
}

#ifdef __AVX__
template<>
inline __m256 lerp<__m256>(const __m256 a, const __m256 b, const float t)
{
	__m256 tVector = _mm256_set1_ps(t);
	__m256 inverseTVector = _mm256_set1_ps(1.0f - t);
	
	return _mm256_add_ps(_mm256_mul_ps(inverseTVector, b), _mm256_mul_ps(tVector, a));
}

inline __m256 lerp(const __m256 a, const __m256 b, const __m256 t)
{
	__m256 one = _mm256_set1_ps(1.0f);
	__m256 inverseTVector = _mm256_sub_ps(one, t);
	
	return _mm256_add_ps(_mm256_mul_ps(inverseTVector, b), _mm256_mul_ps(t, a));
}
#endif

template<typename T>
HOST_DEVICE inline T saturate(const T x)
{
#ifndef __NVCC__
	return std::min(1.0f, (std::max(0.0f, x)));
#else
	return fminf(1.0f, (fmaxf(0.0f, x)));
#endif
}

template<>
HOST_DEVICE inline Math::Vector4 saturate<Math::Vector4>(const Math::Vector4 v)
{
	return {saturate(v.x()), saturate(v.y()), saturate(v.z())};
}


} // namespace Math

#endif // ALGORITHMS_H
