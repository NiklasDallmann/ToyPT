#ifndef MATRIX3D_H
#define MATRIX3D_H

#include <array>
#include <stddef.h>

#include "vector3d.h"

namespace Math
{

class Matrix3D
{
public:
	friend std::ostream &operator<<(std::ostream &stream, const Matrix3D &matrix);
	
	Matrix3D(const Vector3D &x = {}, const Vector3D &y = {}, const Vector3D &z = {});
	
	Matrix3D &transpose();
	Matrix3D transposed() const;
	
	static Matrix3D transposedAdd(const Matrix3D &left, const Matrix3D &right);
	static Matrix3D transposedSubtract(const Matrix3D &left, const Matrix3D &right);
	static Matrix3D transposedMultiply(const Matrix3D &left, const Matrix3D &right);
	
	Matrix3D &operator+=(const Matrix3D &other);
	Matrix3D &operator-=(const Matrix3D &other);
	Matrix3D &operator*=(const Matrix3D &other);
	Matrix3D &operator*=(const double scalar);
	
	Vector3D &operator[](const size_t index);
	Vector3D &operator[](const size_t index) const;
	
private:
	std::array<Vector3D, 4> _vectors;
};

Matrix3D operator+(const Matrix3D &left, const Matrix3D &right);
Matrix3D operator-(const Matrix3D &left, const Matrix3D &right);
Matrix3D operator*(const Matrix3D &left, const Matrix3D &right);

std::ostream &operator<<(std::ostream &stream, const Matrix3D &matrix);

} // namespace Math

#endif // MATRIX3D_H
