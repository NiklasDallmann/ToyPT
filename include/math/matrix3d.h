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
	
	Matrix3D(const Vector3D &a = {}, const Vector3D &b = {}, const Vector3D &c = {}, const Vector3D &d = {});
	
	Matrix3D &transpose();
	Matrix3D transposed() const;
	
	Matrix3D &invert(bool *invertible = nullptr);
	Matrix3D inverted(bool *invertible = nullptr) const;
	
	double determinant3x3() const;
	double determinant() const;
	
	static Matrix3D identityMatrix();
	static Matrix3D rotationMatrixX(const double angle);
	static Matrix3D rotationMatrixY(const double angle);
	static Matrix3D rotationMatrixZ(const double angle);
	
	static Matrix3D transposedAdd(const Matrix3D &left, const Matrix3D &right);
	static Matrix3D transposedSubtract(const Matrix3D &left, const Matrix3D &right);
	static Matrix3D transposedMultiply(const Matrix3D &left, const Matrix3D &right);
	
	Matrix3D &operator+=(const Matrix3D &other);
	Matrix3D &operator-=(const Matrix3D &other);
	Matrix3D &operator*=(const Matrix3D &other);
	Matrix3D &operator*=(const double scalar);
	Vector3D operator*(const Vector3D &vector);
	
	Vector3D &operator[](const size_t index);
	Vector3D &operator[](const size_t index) const;
	
private:
	static constexpr size_t _dimension = 4;
	std::array<Vector3D, 4> _vectors;
};

Matrix3D operator+(const Matrix3D &left, const Matrix3D &right);
Matrix3D operator-(const Matrix3D &left, const Matrix3D &right);
Matrix3D operator*(const Matrix3D &left, const Matrix3D &right);
Matrix3D operator*(const Matrix3D &left, const double right);
Matrix3D operator*(const double left, const Matrix3D &right);

std::ostream &operator<<(std::ostream &stream, const Matrix3D &matrix);

} // namespace Math

#endif // MATRIX3D_H
