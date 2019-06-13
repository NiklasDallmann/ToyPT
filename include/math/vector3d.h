#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <array>
#include <ostream>
#include <stddef.h>

///
/// Contains mathematical primitives.
/// 
/// \since	1.0
///
namespace Math
{

///
/// Implements a four-dimensional vector for 3D space.
/// 
/// \since	1.0
///
class Vector3D
{
public:
	friend float operator*(const Vector3D &left, const Vector3D &right);
	friend std::ostream &operator<<(std::ostream &stream, const Vector3D &vector);
	
	///
	/// Constructs a vector with its coordinates \a x, \a y and \a z.
	/// 
	/// \since	1.0
	///
	Vector3D(const float x = 0, const float y = 0, const float z = 0, const float w = 1);
	
	///
	/// Returns the x coordinate.
	/// 
	/// \sa		setX()
	/// 
	/// \since	1.0
	///
	float x() const;
	
	///
	/// Sets the x coordinate to \a x.
	/// 
	/// \sa		x()
	/// 
	/// \since	1.0
	///
	void setX(const float x);
	
	///
	/// Returns the y coordinate.
	/// 
	/// \sa		setY()
	/// 
	/// \since	1.0
	///
	float y() const;
	
	///
	/// Sets the y coordinate to \a y.
	/// 
	/// \sa		y()
	/// 
	/// \since	1.0
	///
	void setY(const float y);
	
	///
	/// Returns the z coordinate.
	/// 
	/// \sa		setZ()
	/// 
	/// \since	1.0
	///
	float z() const;
	
	///
	/// Sets the z coordinate to \a z.
	/// 
	/// \sa		z()
	/// 
	/// \since	1.0
	///
	void setZ(const float z);
	
	///
	/// Returns the magnitude of the vector.
	/// 
	/// \since	1.0
	///
	float magnitude() const;
	
	///
	/// Normalizes the vector so that it has a magnitude of one and returns a reference to the vector.
	/// 
	/// \sa		normalized()
	/// 
	/// \since	1.0
	///
	Vector3D &normalize();
	
	///
	/// Copies the vector and normalizes it.
	/// 
	/// \sa		normalize()
	/// 
	/// \since	1.0
	///
	Vector3D normalized() const;
	
	///
	/// Returns the cross product of the vector and \a other.
	/// 
	/// \since	1.0
	///
	Vector3D crossProduct(const Vector3D &other) const;
	
	Vector3D coordinateProduct(const Vector3D &other) const;
	
	///
	/// Calculates and returns the cosinus between \a left and \a right.
	/// 
	/// \since	1.0
	///
	float cos(const Vector3D &left, const Vector3D &right);
	
	///
	/// Adds \a other to the vector and returns a reference to it.
	/// 
	/// \since	1.0
	///
	Vector3D &operator+=(const Vector3D &other);
	
	///
	/// Subtracts \a other from the vector and returns a reference to it.
	/// 
	/// \since	1.0
	///
	Vector3D &operator-=(const Vector3D &other);
	
	///
	/// Multiplies the vector by \a scalar and returns a reference to it.
	/// 
	/// \since	1.0
	///
	Vector3D &operator*=(const float scalar);
	
	///
	/// Divides the vector by \a scalar and returns a reference to it.
	/// 
	/// \since	1.0
	///
	Vector3D &operator/=(const float scalar);
	
	Vector3D operator-() const;
	
	///
	/// Returns a reference to the coordinate at \a index.
	/// 
	/// \since	1.0
	///
	float &operator[](const size_t index);
	
	///
	/// Returns the coordinate at \a index.
	/// 
	/// \since	1.0
	///
	float operator[](const size_t index) const;
	
	bool operator==(const Vector3D &other);
	bool operator!=(const Vector3D &other);
	
private:
	static constexpr size_t _dimension = 4;
	alignas (sizeof (float) * _dimension) std::array<float, _dimension> _coordinates;
};

///
/// Adds \a left and \a right and returns the result.
/// 
/// \since	1.0
///
Vector3D operator+(const Vector3D &left, const Vector3D &right);

///
/// Subtracts \a right from \a left and returns the result.
/// 
/// \since	1.0
///
Vector3D operator-(const Vector3D &left, const Vector3D &right);

///
/// Returns the dot product of \a left and \a right.
/// 
/// \since	1.0
///
float operator*(const Vector3D &left, const Vector3D &right);

///
/// Multiplies \a left by \a right and returns the result.
/// 
/// \since	1.0
///
Vector3D operator*(const Vector3D &left, const float right);

///
/// Multiplies \a right by \a left and returns the result.
/// 
/// \since	1.0
///
Vector3D operator*(const float left, const Vector3D right);

///
/// Divides \a left by \a right and returns the result.
/// 
/// \since	1.0
///
Vector3D operator/(const Vector3D &left, const float right);

///
/// Divides \a right by \a left and returns the result.
/// 
/// \since	1.0
///
Vector3D operator/(const float left, const Vector3D right);

///
/// Writes a JSON representation of \a vector to \a stream.
/// 
/// \since	1.0
///
std::ostream &operator<<(std::ostream &stream, const Vector3D &vector);

} // namespace Math

#endif // VECTOR3D_H
