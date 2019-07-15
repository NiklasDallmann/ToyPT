#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <algorithm>

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


} // namespace Math

#endif // ALGORITHMS_H
