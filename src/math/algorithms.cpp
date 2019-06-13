#include <algorithm>

#include "algorithms.h"

namespace Math
{

template<>
bool fuzzyCompareEqual<float>(const float left, const float right)
{
	bool returnValue = false;
	
	returnValue = std::abs(std::abs(left) - std::abs(right)) < epsilon;
	
	return returnValue;
}

}
