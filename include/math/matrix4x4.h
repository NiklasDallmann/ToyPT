#ifndef MATRIX4X4_H
#define MATRIX4X4_H

#include <array>
#include <cmath>
#include <sstream>
#include <stddef.h>
#include <utility>

#include "algorithms.h"
#include "vector4.h"

namespace Math
{

class Matrix4x4
{
public:
	friend std::ostream &operator<<(std::ostream &stream, const Matrix4x4 &matrix);
	
	Matrix4x4(const Vector4 &a = {}, const Vector4 &b = {}, const Vector4 &c = {}, const Vector4 &d = {}) :
		_vectors({a, b, c, d})
	{
	}
	
	Matrix4x4 &transpose()
	{
		for (size_t i = 0; i < (_dimension - 1); i++)
		{
			for (size_t j = i + 1; j < _dimension; j++)
			{
				std::swap((*this)[i][j], (*this)[j][i]);
			}
		}
		
		return *this;
	}
	Matrix4x4 transposed() const
	{
		Matrix4x4 returnValue = *this;
		
		returnValue.transpose();
		
		return returnValue;
	}
	
	Matrix4x4 &invert(bool *invertible = nullptr)
	{
		bool isInvertible = true;
		Matrix4x4 temporary = Matrix4x4::identityMatrix();
		float determinant = this->determinant3x3();
		
		if (determinant == 0.0f)
		{
			goto exit;
		}
		
		// Select pivot elements being not zero
		for (size_t row = 0; row < _dimension; row++)
		{
			if (Math::fuzzyCompareEqual((*this)[row][row], 0.0f))
			{
				// Look for another row
			}
		}
		
		// Subtract multiples of rows
		
		
		isInvertible = true;
		
exit:
		if (invertible != nullptr)
		{
			*invertible = isInvertible;
		}
		
		return *this;
	}
	Matrix4x4 inverted(bool *invertible = nullptr) const
	{
		return {};
	}
	
	float determinant3x3() const
	{
		float returnValue = 0;
		
		returnValue = (
					(*this)[0][0] * (*this)[1][1] * (*this)[2][2] + 
				(*this)[1][0] * (*this)[2][1] * (*this)[0][2] +
				(*this)[2][0] * (*this)[0][1] * (*this)[1][2] -
				(*this)[0][2] * (*this)[1][1] * (*this)[2][0] -
				(*this)[1][2] * (*this)[2][1] * (*this)[0][0] -
				(*this)[2][2] * (*this)[0][1] * (*this)[1][0]
				);
		
		return returnValue;
	}
	float determinant() const
	{
		float returnValue = 0;
		float d0, d1, d2, d3;
		
		float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
		
		a = (*this)[0][0];
		b = (*this)[0][1];
		c = (*this)[0][2];
		d = (*this)[0][3];
		e = (*this)[1][0];
		f = (*this)[1][1];
		g = (*this)[1][2];
		h = (*this)[1][3];
		i = (*this)[2][0];
		j = (*this)[2][1];
		k = (*this)[2][2];
		l = (*this)[2][3];
		m = (*this)[3][0];
		n = (*this)[3][1];
		o = (*this)[3][2];
		p = (*this)[3][3];
		
		d0 = a * (f*k*p + g*l*n + h*j*o - h*k*n - g*j*p - f*l*o);
		
		d1 = b * (e*k*p + g*l*n + h*i*o - h*k*n - g*i*p - e*l*o);
		
		d2 = c * (e*j*p + f*l*n + h*i*n - h*i*n - f*i*p - e*l*n);
		
		d3 = d * (e*j*o + f*k*m + g*i*n - g*j*m - f*i*o - e*k*n);
		
		returnValue = d0 - d1 + d2 - d3;
		
		return returnValue;
	}
	
	static Matrix4x4 identityMatrix()
	{
		return Matrix4x4{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};
	}
	static Matrix4x4 rotationMatrixX(const float angle)
	{
		float cos = std::cos(angle);
		float sin = std::sin(angle);
		
		Matrix4x4 returnValue{
			{1,		0,		0},
			{0,		cos,	-sin},
			{0,		sin,	cos}
		};
		
		return returnValue;
	}
	static Matrix4x4 rotationMatrixY(const float angle)
	{
		float cos = std::cos(angle);
		float sin = std::sin(angle);
		
		Matrix4x4 returnValue{
			{cos,		0,		sin},
			{0,			1,		0},
			{-sin,		0,		cos}
		};
		
		return returnValue;
	}
	static Matrix4x4 rotationMatrixZ(const float angle)
	{
		float cos = std::cos(angle);
		float sin = std::sin(angle);
		
		Matrix4x4 returnValue{
			{cos,		-sin,		0},
			{sin,		cos,		0},
			{0,			0,			1}
		};
		
		return returnValue;
	}
	
	static Matrix4x4 transposedAdd(const Matrix4x4 &left, const Matrix4x4 &right)
	{
		Matrix4x4 returnValue;
		
		for (size_t i = 0; i < _dimension; i++)
		{
			returnValue[i] = left[i] + right[i];
		}
		
		return returnValue;
	}
	static Matrix4x4 transposedSubtract(const Matrix4x4 &left, const Matrix4x4 &right)
	{
		Matrix4x4 returnValue;
		
		for (size_t i = 0; i < _dimension; i++)
		{
			returnValue[i] = left[i] - right[i];
		}
		
		return returnValue;
	}
	static Matrix4x4 transposedMultiply(const Matrix4x4 &left, const Matrix4x4 &right)
	{
		Matrix4x4 returnValue;
		
		for (size_t i = 0; i < _dimension; i++)
		{
			for (size_t j = 0; j < _dimension; j++)
			{
				returnValue[i][j] = left[i] * right[j];
			}
		}
		
		return returnValue;
	}
	
	Matrix4x4 &operator+=(const Matrix4x4 &other)
	{
		Matrix4x4 transposedOther = other.transposed();
		
		*this = Matrix4x4::transposedAdd(*this, transposedOther);
		
		return *this;
	}
	
	Matrix4x4 &operator-=(const Matrix4x4 &other)
	{
		Matrix4x4 transposedOther = other.transposed();
		
		*this = Matrix4x4::transposedSubtract(*this, transposedOther);
		
		return *this;
	}
	
	Matrix4x4 &operator*=(const Matrix4x4 &other)
	{
		Matrix4x4 transposedOther = other.transposed();
		
		*this = Matrix4x4::transposedMultiply(*this, transposedOther);
		
		return *this;
	}
	
	Matrix4x4 &operator*=(const float scalar)
	{
		for (Vector4 &vector : this->_vectors)
		{
			vector *= scalar;
		}
		
		return *this;
	}
	
	Vector4 &operator[](const size_t index)
	{
		return this->_vectors[index];
	}
	
	const Vector4 &operator[](const size_t index) const
	{
		return this->_vectors[index];
	}
	
private:
	static constexpr size_t _dimension = 4;
	alignas (Vector4) std::array<Vector4, _dimension> _vectors;
};

inline Matrix4x4 operator+(const Matrix4x4 &left, const Matrix4x4 &right)
{
	Matrix4x4 returnValue = left;
	
	returnValue += right;
	
	return returnValue;
}

inline Matrix4x4 operator-(const Matrix4x4 &left, const Matrix4x4 &right)
{
	Matrix4x4 returnValue = left;
	
	returnValue -= right;
	
	return returnValue;
}

inline Matrix4x4 operator*(const Matrix4x4 &left, const Matrix4x4 &right)
{
	Matrix4x4 returnValue = left;
	
	returnValue *= right;
	
	return returnValue;
}

inline Matrix4x4 operator*(const Matrix4x4 &left, const float right)
{
	Matrix4x4 returnValue = left;
	returnValue *= right;
	
	return returnValue;
}

inline Matrix4x4 operator*(const float left, const Matrix4x4 &right)
{
	return right * left;
}

inline Vector4 operator*(const Matrix4x4 &left, const Vector4 &right)
{
	Vector4 returnValue;
	
	for (size_t i = 0; i < 3; i++)
	{
		returnValue[i] = left[i] * right;
	}
	
	return returnValue;
}

inline std::ostream &operator<<(std::ostream &stream, const Matrix4x4 &matrix)
{
	std::stringstream stringStream;
	
	stringStream << "[" << matrix[0] << ", " << matrix[1] << ", " << matrix[2] << "]";
	
	stream << stringStream.str();
	
	return stream;
}

} // namespace Math

#endif // MATRIX4X4_H
