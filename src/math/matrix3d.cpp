#include "matrix3d.h"

#include <cmath>
#include <sstream>
#include <utility>

namespace Math
{

Matrix3D::Matrix3D(const Vector3D &x, const Vector3D &y, const Vector3D &z) :
	_vectors({x, y, z, Vector3D()})
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

Matrix3D &Matrix3D::invert()
{
	// FIXME implement
	return *this;
}

Matrix3D Matrix3D::inverted() const
{
	Matrix3D returnValue = Matrix3D::identityMatrix();
	Matrix3D copy = *this;
	
	// Gauß-Jordan
	for (size_t i = 0; i < _dimension; i++)
	{
		double factor = copy[i][i];
		copy[i] /= factor;
		returnValue[i] *= factor;
		
		for (size_t j = i; j < _dimension; j++)
		{
			copy[j] -= copy[j][i] * copy[i];
			returnValue[j] -= returnValue[j][i] * copy[i];
		}
	}
	
	return returnValue;
}

double Matrix3D::determinant() const
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

Matrix3D Matrix3D::identityMatrix()
{
	return Matrix3D{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1}
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
	
	for (size_t i = 0; i < 3; i++)
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

Vector3D Matrix3D::operator*(const Vector3D &vector)
{
	Vector3D returnValue;
	
	for (size_t i = 0; i < 3; i++)
	{
		returnValue[i] = (*this)[i] * vector;
	}
	
	return returnValue;
}

Vector3D &Matrix3D::operator[](const size_t index)
{
	return this->_vectors[index];
}

Vector3D &Matrix3D::operator[](const size_t index) const
{
	return const_cast<Vector3D &>(this->_vectors[index]);
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

}
