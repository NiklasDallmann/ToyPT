#include <cmath>
#include <sstream>
#include <utility>

#include "algorithms.h"
#include "matrix3d.h"

namespace Math
{

Matrix3D::Matrix3D(const Vector3D &a, const Vector3D &b, const Vector3D &c, const Vector3D &d) :
	_vectors({a, b, c, d})
{
}

Matrix3D &Matrix3D::transpose()
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

Matrix3D Matrix3D::transposed() const
{
	Matrix3D returnValue = *this;
	
	returnValue.transpose();
	
	return returnValue;
}

Matrix3D &Matrix3D::invert(bool *invertible)
{
	bool isInvertible = true;
	Matrix3D temporary = Matrix3D::identityMatrix();
	double determinant = this->determinant3x3();
	
	if (determinant == 0.0)
	{
		goto exit;
	}
	
	// Select pivot elements being not zero
	for (size_t row = 0; row < _dimension; row++)
	{
		if (Math::fuzzyCompareEqual((*this)[row][row], 0.0))
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

Matrix3D Matrix3D::inverted(bool *invertible) const
{
	return {};
}

double Matrix3D::determinant3x3() const
{
	double returnValue = 0;
	
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

double Matrix3D::determinant() const
{
	double returnValue = 0;
	double d0, d1, d2, d3;
	
	double a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
	
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

Matrix3D Matrix3D::identityMatrix()
{
	return Matrix3D{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1}
	};
}

Matrix3D Matrix3D::rotationMatrixX(const double angle)
{
	double cos = std::cos(angle);
	double sin = std::sin(angle);
	
	Matrix3D returnValue{
		{1,		0,		0},
		{0,		cos,	-sin},
		{0,		sin,	cos}
	};
	
	return returnValue;
}

Matrix3D Matrix3D::rotationMatrixY(const double angle)
{
	double cos = std::cos(angle);
	double sin = std::sin(angle);
	
	Matrix3D returnValue{
		{cos,		0,		sin},
		{0,			1,		0},
		{-sin,		0,		cos}
	};
	
	return returnValue;
}

Matrix3D Matrix3D::rotationMatrixZ(const double angle)
{
	double cos = std::cos(angle);
	double sin = std::sin(angle);
	
	Matrix3D returnValue{
		{cos,		-sin,		0},
		{sin,		cos,		0},
		{0,			0,			1}
	};
	
	return returnValue;
}

Matrix3D Matrix3D::transposedAdd(const Matrix3D &left, const Matrix3D &right)
{
	Matrix3D returnValue;
	
	for (size_t i = 0; i < _dimension; i++)
	{
		returnValue[i] = left[i] + right[i];
	}
	
	return returnValue;
}

Matrix3D Matrix3D::transposedSubtract(const Matrix3D &left, const Matrix3D &right)
{
	Matrix3D returnValue;
	
	for (size_t i = 0; i < _dimension; i++)
	{
		returnValue[i] = left[i] - right[i];
	}
	
	return returnValue;
}

Matrix3D Matrix3D::transposedMultiply(const Matrix3D &left, const Matrix3D &right)
{
	Matrix3D returnValue;
	
	for (size_t i = 0; i < _dimension; i++)
	{
		for (size_t j = 0; j < _dimension; j++)
		{
			returnValue[i][j] = left[i] * right[j];
		}
	}
	
	return returnValue;
}

Matrix3D &Matrix3D::operator+=(const Matrix3D &other)
{
	Matrix3D transposedOther = other.transposed();
	
	*this = Matrix3D::transposedAdd(*this, transposedOther);
	
	return *this;
}

Matrix3D &Matrix3D::operator-=(const Matrix3D &other)
{
	Matrix3D transposedOther = other.transposed();
	
	*this = Matrix3D::transposedSubtract(*this, transposedOther);
	
	return *this;
}

Matrix3D &Matrix3D::operator*=(const Matrix3D &other)
{
	Matrix3D transposedOther = other.transposed();
	
	*this = Matrix3D::transposedMultiply(*this, transposedOther);
	
	return *this;
}

Matrix3D &Matrix3D::operator*=(const double scalar)
{
	for (Vector3D &vector : this->_vectors)
	{
		vector *= scalar;
	}
	
	return *this;
}

Vector3D &Matrix3D::operator[](const size_t index)
{
	return this->_vectors[index];
}

const Vector3D &Matrix3D::operator[](const size_t index) const
{
	return this->_vectors[index];
}

std::ostream &operator<<(std::ostream &stream, const Matrix3D &matrix)
{
	std::stringstream stringStream;
	
	stringStream << "[" << matrix[0] << ", " << matrix[1] << ", " << matrix[2] << "]";
	
	stream << stringStream.str();
	
	return stream;
}

Matrix3D operator+(const Matrix3D &left, const Matrix3D &right)
{
	Matrix3D returnValue = left;
	
	returnValue += right;
	
	return returnValue;
}

Matrix3D operator-(const Matrix3D &left, const Matrix3D &right)
{
	Matrix3D returnValue = left;
	
	returnValue -= right;
	
	return returnValue;
}

Matrix3D operator*(const Matrix3D &left, const Matrix3D &right)
{
	Matrix3D returnValue = left;
	
	returnValue *= right;
	
	return returnValue;
}

Matrix3D operator*(const Matrix3D &left, const double right)
{
	Matrix3D returnValue = left;
	returnValue *= right;
	
	return returnValue;
}

Matrix3D operator*(const double left, const Matrix3D &right)
{
	return right * left;
}

Vector3D operator*(const Matrix3D &left, const Vector3D &right)
{
	Vector3D returnValue;
	
	for (size_t i = 0; i < 3; i++)
	{
		returnValue[i] = left[i] * right;
	}
	
	return returnValue;
}

}
