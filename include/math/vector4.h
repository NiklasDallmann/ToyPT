#ifndef VECTOR4_H
#define VECTOR4_H

#include <array>
#include <cmath>
#include <ostream>
#include <sstream>
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
class Vector4
{
public:
	friend float operator*(const Vector4 &left, const Vector4 &right);
	friend std::ostream &operator<<(std::ostream &stream, const Vector4 &vector);
	
	///
	/// Constructs a vector with its coordinates \a x, \a y and \a z.
	/// 
	/// \since	1.0
	///
	Vector4(const float x = 0, const float y = 0, const float z = 0, const float w = 1) :
		_coordinates({x, y, z, w})
	{
	}
	
	///
	/// Returns the x coordinate.
	/// 
	/// \sa		setX()
	/// 
	/// \since	1.0
	///
	float x() const
	{
		return this->_coordinates[0];
	}
	
	///
	/// Sets the x coordinate to \a x.
	/// 
	/// \sa		x()
	/// 
	/// \since	1.0
	///
	void setX(const float x)
	{
		this->_coordinates[0] = x;
	}
	
	///
	/// Returns the y coordinate.
	/// 
	/// \sa		setY()
	/// 
	/// \since	1.0
	///
	float y() const
	{
		return this->_coordinates[1];
	}
	
	///
	/// Sets the y coordinate to \a y.
	/// 
	/// \sa		y()
	/// 
	/// \since	1.0
	///
	void setY(const float y)
	{
		this->_coordinates[1] = y;
	}
	
	///
	/// Returns the z coordinate.
	/// 
	/// \sa		setZ()
	/// 
	/// \since	1.0
	///
	float z() const
	{
		return this->_coordinates[2];
	}
	
	///
	/// Sets the z coordinate to \a z.
	/// 
	/// \sa		z()
	/// 
	/// \since	1.0
	///
	void setZ(const float z)
	{
		this->_coordinates[2] = z;
	}
	
	///
	/// Returns the w coordinate.
	/// 
	/// \sa		setW()
	/// 
	/// \since	1.0
	///
	float w() const
	{
		return this->_coordinates[3];
	}
	
	///
	/// Sets the w coordinate to \a w.
	/// 
	/// \sa		w()
	/// 
	/// \since	1.0
	///
	void setW(const float w)
	{
		this->_coordinates[3] = w;
	}
	
	///
	/// Returns the magnitude of the vector.
	/// 
	/// \since	1.0
	///
	float magnitude() const
	{
		return std::pow((std::pow(this->_coordinates[0], 2.0f) + std::pow(this->_coordinates[1], 2.0f) + std::pow(this->_coordinates[2], 2.0f)), 0.5f);
	}
	
	///
	/// Normalizes the vector so that it has a magnitude of one and returns a reference to the vector.
	/// 
	/// \sa		normalized()
	/// 
	/// \since	1.0
	///
	Vector4 &normalize()
	{
		float magnitude = this->magnitude();
		
		this->_coordinates[0] /= magnitude;
		this->_coordinates[1] /= magnitude;
		this->_coordinates[2] /= magnitude;
		
		return *this;
	}
	
	///
	/// Copies the vector and normalizes it.
	/// 
	/// \sa		normalize()
	/// 
	/// \since	1.0
	///
	Vector4 normalized() const
	{
		Vector4 returnValue = *this;
		
		returnValue.normalize();
		
		return returnValue;
	}
	
	///
	/// Returns the cross product of the vector and \a other.
	/// 
	/// \since	1.0
	///
	Vector4 crossProduct(const Vector4 &other) const
	{
		Vector4 returnValue;
		
		returnValue[0] = (*this)[1] * other[2] - (*this)[2] * other[1];
		returnValue[1] = (*this)[2] * other[0] - (*this)[0] * other[2];
		returnValue[2] = (*this)[0] * other[1] - (*this)[1] * other[0];
		
		return returnValue;
	}
	
	Vector4 coordinateProduct(const Vector4 &other) const
	{
		Vector4 returnValue;
		
		returnValue[0] = (*this)[0] * other[0];
		returnValue[1] = (*this)[1] * other[1];
		returnValue[2] = (*this)[2] * other[2];
		
		return returnValue;
	}
	
	///
	/// Calculates and returns the cosinus between \a left and \a right.
	/// 
	/// \since	1.0
	///
	float cos(const Vector4 &left, const Vector4 &right)
	{
		float returnValue = 0;
		
		returnValue = ((left * right) / (left.magnitude() * right.magnitude()));
		
		return returnValue;
	}
	
	///
	/// Adds \a other to the vector and returns a reference to it.
	/// 
	/// \since	1.0
	///
	Vector4 &operator+=(const Vector4 &other)
	{
		this->_coordinates[0] += other._coordinates[0];
		this->_coordinates[1] += other._coordinates[1];
		this->_coordinates[2] += other._coordinates[2];
		
		return *this;
	}
	
	///
	/// Subtracts \a other from the vector and returns a reference to it.
	/// 
	/// \since	1.0
	///
	Vector4 &operator-=(const Vector4 &other)
	{
		this->_coordinates[0] -= other._coordinates[0];
		this->_coordinates[1] -= other._coordinates[1];
		this->_coordinates[2] -= other._coordinates[2];
		
		return *this;
	}
	
	///
	/// Multiplies the vector by \a scalar and returns a reference to it.
	/// 
	/// \since	1.0
	///
	Vector4 &operator*=(const float scalar)
	{
		this->_coordinates[0] *= scalar;
		this->_coordinates[1] *= scalar;
		this->_coordinates[2] *= scalar;
		
		return *this;
	}
	
	///
	/// Divides the vector by \a scalar and returns a reference to it.
	/// 
	/// \since	1.0
	///
	Vector4 &operator/=(const float scalar)
	{
		this->_coordinates[0] /= scalar;
		this->_coordinates[1] /= scalar;
		this->_coordinates[2] /= scalar;
		
		return *this;
	}
	
	Vector4 operator-() const
	{
		return *this * (-1.0);
	}
	
	///
	/// Returns a reference to the coordinate at \a index.
	/// 
	/// \since	1.0
	///
	float &operator[](const size_t index)
	{
		return this->_coordinates[index];
	}
	
	///
	/// Returns the coordinate at \a index.
	/// 
	/// \since	1.0
	///
	float operator[](const size_t index) const
	{
		return this->_coordinates[index];
	}
	
	bool operator==(const Vector4 &other)
	{
		return (this->_coordinates == other._coordinates);
	}
	
	bool operator!=(const Vector4 &other)
	{
		return !(*this == other);
	}
	
private:
	static constexpr size_t _dimension = 4;
	alignas (sizeof (float) * _dimension) std::array<float, _dimension> _coordinates;
};

///
/// Adds \a left and \a right and returns the result.
/// 
/// \since	1.0
///
inline Vector4 operator+(const Vector4 &left, const Vector4 &right)
{
	Vector4 returnValue = left;
	
	returnValue += right;
	
	return returnValue;
}

///
/// Subtracts \a right from \a left and returns the result.
/// 
/// \since	1.0
///
inline Vector4 operator-(const Vector4 &left, const Vector4 &right)
{
	Vector4 returnValue = left;
	
	returnValue -= right;
	
	return returnValue;
}

///
/// Returns the dot product of \a left and \a right.
/// 
/// \since	1.0
///
inline float operator*(const Vector4 &left, const Vector4 &right)
{
	float returnValue = 0;
	
	returnValue += (left._coordinates[0] * right._coordinates[0]);
	returnValue += (left._coordinates[1] * right._coordinates[1]);
	returnValue += (left._coordinates[2] * right._coordinates[2]);
	
	return returnValue;
}

///
/// Multiplies \a left by \a right and returns the result.
/// 
/// \since	1.0
///
inline Vector4 operator*(const Vector4 &left, const float right)
{
	Vector4 returnValue = left;
	
	returnValue *= right;
	
	return returnValue;
}

///
/// Multiplies \a right by \a left and returns the result.
/// 
/// \since	1.0
///
inline Vector4 operator*(const float left, const Vector4 right)
{
	return right * left;
}

///
/// Divides \a left by \a right and returns the result.
/// 
/// \since	1.0
///
inline Vector4 operator/(const Vector4 &left, const float right)
{
	Vector4 returnValue = left;
	
	returnValue /= right;
	
	return returnValue;
}

///
/// Divides \a right by \a left and returns the result.
/// 
/// \since	1.0
///
inline Vector4 operator/(const float left, const Vector4 right)
{
	return right / left;
}

///
/// Writes a JSON representation of \a vector to \a stream.
/// 
/// \since	1.0
///
inline std::ostream &operator<<(std::ostream &stream, const Vector4 &vector)
{
	std::stringstream stringStream;
	
	stringStream << "{\"x\": " << vector._coordinates[0] << ", \"y\": " << vector._coordinates[1] << ", \"z\": " << vector._coordinates[2] << "}";
	
	stream << stringStream.str();
	
	return stream;
}

} // namespace Math

#endif // VECTOR4_H
