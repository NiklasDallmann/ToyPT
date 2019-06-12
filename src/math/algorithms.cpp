#include <algorithm>

#include "algorithms.h"

namespace Math
{

template<>
bool fuzzyCompareEqual<double>(const double left, const double right)
{
	bool returnValue = false;
	
	returnValue = std::abs(std::abs(left) - std::abs(right)) < epsilon;
	
	return returnValue;
}

}
