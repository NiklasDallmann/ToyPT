#include "matrix3d.h"

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
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = i + 1; j < 3; j++)
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
	
	for (size_t i = 0; i < 3; i++)
	{
		returnValue[i] = left[i] - right[i];
	}
	
	return returnValue;
}

Matrix3D Matrix3D::transposedMultiply(const Matrix3D &left, const Matrix3D &right)
{
	Matrix3D returnValue;
	
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
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
