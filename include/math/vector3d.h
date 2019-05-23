#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <array>
#include <ostream>
#include <stddef.h>

namespace Math
{

///
/// Implements a three-dimensional vector.
/// 
/// For cache alignment a forth coordinate is used in the internal storage. It is ignored by any operation.
/// 
/// \since	1.0
///
class Vector3D
{
public:
	friend double operator*(const Vector3D &left, const Vector3D &right);
	friend std::ostream &operator<<(std::ostream &stream, const Vector3D &vector);
	
	///
	/// Constructs a vector with its coordinates \a x, \a y and \a z.
	/// 
	/// \since	1.0
	///
	Vector3D(const double x = 0, const double y = 0, const double z = 0);
	
	///
	/// Returns the x coordinate.
	/// 
	/// \sa		setX()
	/// 
	/// \since	1.0
	///
	double x() const;
	
	///
	/// Sets the x coordinate to \a x.
	/// 
	/// \sa		x()
	/// 
	/// \since	1.0
	///
	void setX(const double x);
	
	///
	/// Returns the y coordinate.
	/// 
	/// \sa		setY()
	/// 
	/// \since	1.0
	///
	double y() const;
	
	///
	/// Sets the y coordinate to \a y.
	/// 
	/// \sa		y()
	/// 
	/// \since	1.0
	///
	void setY(const double y);
	
	///
	/// Returns the z coordinate.
	/// 
	/// \sa		setZ()
	/// 
	/// \since	1.0
	///
	double z() const;
	
	///
	/// Sets the z coordinate to \a z.
	/// 
	/// \sa		z()
	/// 
	/// \since	1.0
	///
	void setZ(const double z);
	
	///
	/// Returns the magnitude of the vector.
	/// 
	/// \since	1.0
	///
	double magnitude() const;
	
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
	/// Calculates the cosinus between \a left and \a right.
	/// 
	/// \since	1.0
	///
	double cos(const Vector3D &left, const Vector3D &right);
	
	Vector3D &operator+=(const Vector3D &other);
	Vector3D &operator-=(const Vector3D &other);
	Vector3D &operator*=(const double scalar);
	Vector3D &operator/=(const double scalar);
	
	double &operator[](const size_t index);
	double operator[](const size_t index) const;
	
private:
	std::array<double, 4> _coordinates;
};

Vector3D operator+(const Vector3D &left, const Vector3D &right);
Vector3D operator-(const Vector3D &left, const Vector3D &right);
double operator*(const Vector3D &left, const Vector3D &right);
Vector3D operator*(const Vector3D &left, const double right);
Vector3D operator*(const double left, const Vector3D right);

std::ostream &operator<<(std::ostream &stream, const Vector3D &vector);

} // namespace Math

#endif // VECTOR3D_H
