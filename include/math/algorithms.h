#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <algorithm>

namespace Math
{

constexpr float epsilon = 1e-6;

template <typename T>
bool fuzzyCompareEqual(const T left, const T right);

template<>
bool fuzzyCompareEqual<float>(const float left, const float right)
{
	bool returnValue = false;
	
	returnValue = std::abs(std::abs(left) - std::abs(right)) < epsilon;
	
	return returnValue;
}


} // namespace Math

#endif // ALGORITHMS_H
