#ifndef RAY_H
#define RAY_H

#include <math/vector4.h>

namespace ToyPT
{
namespace Rendering
{

struct Ray
{
	Math::Vector4 origin;
	Math::Vector4 direction;
};

} // namespace Rendering
} // namespace ToyPT

#endif // RAY_H
