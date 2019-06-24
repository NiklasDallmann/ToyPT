#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <array>
#include <stddef.h>
#include <vector4.h>

namespace Rendering
{

class Triangle
{
public:
	Triangle(const std::array<Math::Vector4, 3> &vertices);
	
	Math::Vector4 normal() const;
	
	std::array<Math::Vector4, 3> &vertices();
	const std::array<Math::Vector4, 3> &vertices() const;
	
	Math::Vector4 &operator[](const size_t index);
	const Math::Vector4 &operator[](const size_t index) const;
	
private:
	std::array<Math::Vector4, 3> _vertices;
};

} // namespace Rendering

#endif // TRIANGLE_H
