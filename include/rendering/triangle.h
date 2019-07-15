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
	Triangle(const std::array<Math::Vector4, 3> &vertices, const std::array<Math::Vector4, 3> &normals = {});
	
	Math::Vector4 normal() const;
	
	std::array<Math::Vector4, 3> &vertices();
	const std::array<Math::Vector4, 3> &vertices() const;
	std::array<Math::Vector4, 3> &normals();
	const std::array<Math::Vector4, 3> &normals() const;
	
	Math::Vector4 &operator[](const size_t index);
	const Math::Vector4 &operator[](const size_t index) const;
	
private:
	std::array<Math::Vector4, 3> _vertices;
	std::array<Math::Vector4, 3> _normals;
	
};

inline std::ostream &operator<<(std::ostream &stream, const Triangle &triangle)
{
	std::stringstream stringStream;
	
	stringStream << "{" << triangle[0] << ", " << triangle[1] << ", " << triangle[2] << "}";
	
	stream << stringStream.str();
	
	return stream;
}

} // namespace Rendering

#endif // TRIANGLE_H
