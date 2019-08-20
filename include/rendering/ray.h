#ifndef RAY_H
#define RAY_H

#include <vector4.h>

namespace Rendering
{

struct Ray
{
	Math::Vector4 origin;
	Math::Vector4 direction;
};

} // namespace Rendering

#endif // RAY_H
