#ifndef VECTOR4_H
#define VECTOR4_H

#include <array>
#include <cmath>
#include <emmintrin.h>
#include <ostream>
#include <sstream>
#include <stddef.h>
#include <utility/globals.h>

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
	friend HOST_DEVICE Vector4 operator*(const Vector4 &left, const Vector4 &right);
#ifndef __NVCC__
	friend std::ostream &operator<<(std::ostream &stream, const Vector4 &vector);
#endif
	
	///
	/// Constructs a vector with its coordinates \a x, \a y and \a z.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE Vector4(const float x = 0, const float y = 0, const float z = 0, const float w = 0)
	{
#ifdef __AVX__
		this->_coordinates = __m128{x, y, z, w};
#else
		this->_coordinates[0] = x;
		this->_coordinates[1] = y;
		this->_coordinates[2] = z;
		this->_coordinates[3] = w;
#endif
	}
	
	///
	/// Returns the x coordinate.
	/// 
	/// \sa		setX()
	/// 
	/// \since	1.0
	///
	HOST_DEVICE float x() const
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
	HOST_DEVICE void setX(const float x)
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
	HOST_DEVICE float y() const
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
	HOST_DEVICE void setY(const float y)
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
	HOST_DEVICE float z() const
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
	HOST_DEVICE void setZ(const float z)
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
	HOST_DEVICE float w() const
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
	HOST_DEVICE void setW(const float w)
	{
		this->_coordinates[3] = w;
	}
	
	///
	/// Returns the magnitude of the vector.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE float magnitude() const
	{
#ifndef __NVCC__
		return std::pow((std::pow(this->_coordinates[0], 2.0f) + std::pow(this->_coordinates[1], 2.0f) + std::pow(this->_coordinates[2], 2.0f)), 0.5f);
#else
		return powf((powf(this->_coordinates[0], 2.0f) + powf(this->_coordinates[1], 2.0f) + powf(this->_coordinates[2], 2.0f)), 0.5f);
#endif
	}
	
	///
	/// Normalizes the vector so that it has a magnitude of one and returns a reference to the vector.
	/// 
	/// \sa		normalized()
	/// 
	/// \since	1.0
	///
	HOST_DEVICE Vector4 &normalize()
	{
#ifdef __AVX__
		__m128 magnitude = _mm_set1_ps(this->magnitude());
		this->_coordinates = _mm_div_ps(this->_coordinates, magnitude);
#else
		float magnitude = this->magnitude();
		(*this) /= magnitude;
#endif
		
		return *this;
	}
	
	///
	/// Copies the vector and normalizes it.
	/// 
	/// \sa		normalize()
	/// 
	/// \since	1.0
	///
	HOST_DEVICE Vector4 normalized() const
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
	HOST_DEVICE Vector4 crossProduct(const Vector4 &other) const
	{
		Vector4 returnValue;
		
		returnValue[0] = (*this)[1] * other[2] - (*this)[2] * other[1];
		returnValue[1] = (*this)[2] * other[0] - (*this)[0] * other[2];
		returnValue[2] = (*this)[0] * other[1] - (*this)[1] * other[0];
		
		return returnValue;
	}
	
	HOST_DEVICE float dotProduct(const Vector4 &other) const
	{
		float returnValue = 0;
		
#ifdef __AVX__
		__m128 temporary = _mm_mul_ps(this->_coordinates, other._coordinates);
		returnValue = temporary[0] + temporary[1] + temporary[2];
#else
		Vector4 temporary = (*this) * other;
		returnValue = temporary[0] + temporary[1] + temporary[2];
#endif
		
		return returnValue;
	}
	
	///
	/// Calculates and returns the cosinus between \a left and \a right.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE float cos(const Vector4 &left, const Vector4 &right)
	{
		float returnValue = 0;
		
		returnValue = ((left.dotProduct(right)) / (left.magnitude() *right.magnitude()));
		
		return returnValue;
	}
	
	///
	/// Adds \a other to the vector and returns a reference to it.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE Vector4 &operator+=(const Vector4 &other)
	{
#ifdef __AVX__
		this->_coordinates = _mm_add_ps(this->_coordinates, other._coordinates);
#else
		this->_coordinates[0] += other._coordinates[0];
		this->_coordinates[1] += other._coordinates[1];
		this->_coordinates[2] += other._coordinates[2];
		this->_coordinates[3] += other._coordinates[3];
#endif
		
		return *this;
	}
	
	///
	/// Subtracts \a other from the vector and returns a reference to it.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE Vector4 &operator-=(const Vector4 &other)
	{
#ifdef __AVX__
		this->_coordinates = _mm_sub_ps(this->_coordinates, other._coordinates);
#else
		this->_coordinates[0] -= other._coordinates[0];
		this->_coordinates[1] -= other._coordinates[1];
		this->_coordinates[2] -= other._coordinates[2];
		this->_coordinates[3] -= other._coordinates[3];
#endif
		
		return *this;
	}
	
	HOST_DEVICE Vector4 &operator*=(const Vector4 &other)
	{
#ifdef __AVX__
		this->_coordinates = _mm_mul_ps(this->_coordinates, other._coordinates);
#else
		this->_coordinates[0] *= other._coordinates[0];
		this->_coordinates[1] *= other._coordinates[1];
		this->_coordinates[2] *= other._coordinates[2];
		this->_coordinates[3] *= other._coordinates[3];
#endif
		
		return *this;
	}
	
	///
	/// Multiplies the vector by \a scalar and returns a reference to it.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE Vector4 &operator*=(const float scalar)
	{
#ifdef __AVX__
		__m128 scalarVector = _mm_set1_ps(scalar);
		this->_coordinates = _mm_mul_ps(this->_coordinates, scalarVector);
#else
		this->_coordinates[0] += scalar;
		this->_coordinates[1] += scalar;
		this->_coordinates[2] += scalar;
		this->_coordinates[3] += scalar;
#endif
		
		return *this;
	}
	
	///
	/// Divides the vector by \a scalar and returns a reference to it.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE Vector4 &operator/=(const float scalar)
	{
#ifdef __AVX__
		__m128 scalarVector = _mm_set1_ps(scalar);
		this->_coordinates = _mm_div_ps(this->_coordinates, scalarVector);
#else
		this->_coordinates[0] /= scalar;
		this->_coordinates[1] /= scalar;
		this->_coordinates[2] /= scalar;
		this->_coordinates[3] /= scalar;
#endif
		
		return *this;
	}
	
	HOST_DEVICE Vector4 operator-() const
	{
		return Vector4{-1.0f, -1.0f, -1.0f} * (*this);
	}
	
	///
	/// Returns a reference to the coordinate at \a index.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE float &operator[](const size_t index)
	{
		return this->_coordinates[index];
	}
	
	///
	/// Returns the coordinate at \a index.
	/// 
	/// \since	1.0
	///
	HOST_DEVICE float operator[](const size_t index) const
	{
		return this->_coordinates[index];
	}
	
#ifdef __NVCC__
	HOST_DEVICE float *data()
	{
		return &this->_coordinates[0];
	}
#endif
	
	HOST_DEVICE bool operator==(const Vector4 &other)
	{
		bool returnValue = true;
		
#ifdef __AVX__
		__m128 resultVector = _mm_cmpeq_ps(this->_coordinates, other._coordinates);
		returnValue = bool(resultVector[0]);
#else
		returnValue =
				((*this)[0] == other[0]) &
				((*this)[1] == other[1]) &
				((*this)[2] == other[2]) &
				((*this)[3] == other[3]);
#endif
		
		return returnValue;
	}
	
	HOST_DEVICE bool operator!=(const Vector4 &other)
	{
		return !(*this == other);
	}
	
private:
	static constexpr size_t _dimension = 4;
	
#ifdef __AVX__
	__m128 _coordinates;
#else
	alignas (sizeof (float) * _dimension) float _coordinates[_dimension];
#endif
};

///
/// Adds \a left and \a right and returns the result.
/// 
/// \since	1.0
///
HOST_DEVICE inline Vector4 operator+(const Vector4 &left, const Vector4 &right)
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
HOST_DEVICE inline Vector4 operator-(const Vector4 &left, const Vector4 &right)
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
HOST_DEVICE inline Vector4 operator*(const Vector4 &left, const Vector4 &right)
{
	Vector4 returnValue = left;
	
	returnValue *= right;
	
	return returnValue;
}

///
/// Multiplies \a left by \a right and returns the result.
/// 
/// \since	1.0
///
HOST_DEVICE inline Vector4 operator*(const Vector4 &left, const float right)
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
HOST_DEVICE inline Vector4 operator*(const float left, const Vector4 right)
{
	return right * left;
}

///
/// Divides \a left by \a right and returns the result.
/// 
/// \since	1.0
///
HOST_DEVICE inline Vector4 operator/(const Vector4 &left, const float right)
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
HOST_DEVICE inline Vector4 operator/(const float left, const Vector4 right)
{
	return right / left;
}

#ifndef __NVCC__
///
/// Writes a JSON representation of \a vector to \a stream.
/// 
/// \since	1.0
///
inline std::ostream &operator<<(std::ostream &stream, const Vector4 &vector)
{
	std::stringstream stringStream;
	
	stringStream << "{\"x\": " << vector[0] << ", \"y\": " << vector[1] << ", \"z\": " << vector[2] << "}";
	
	stream << stringStream.str();
	
	return stream;
}
#endif

} // namespace Math

#endif // VECTOR4_H
