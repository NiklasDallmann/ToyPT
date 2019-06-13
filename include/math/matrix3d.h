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
	
	float determinant3x3() const;
	float determinant() const;
	
	static Matrix3D identityMatrix();
	static Matrix3D rotationMatrixX(const float angle);
	static Matrix3D rotationMatrixY(const float angle);
	static Matrix3D rotationMatrixZ(const float angle);
	
	static Matrix3D transposedAdd(const Matrix3D &left, const Matrix3D &right);
	static Matrix3D transposedSubtract(const Matrix3D &left, const Matrix3D &right);
	static Matrix3D transposedMultiply(const Matrix3D &left, const Matrix3D &right);
	
	Matrix3D &operator+=(const Matrix3D &other);
	Matrix3D &operator-=(const Matrix3D &other);
	Matrix3D &operator*=(const Matrix3D &other);
	Matrix3D &operator*=(const float scalar);
	
	Vector3D &operator[](const size_t index);
	const Vector3D &operator[](const size_t index) const;
	
private:
	static constexpr size_t _dimension = 4;
	alignas (Vector3D) std::array<Vector3D, _dimension> _vectors;
};

Matrix3D operator+(const Matrix3D &left, const Matrix3D &right);
Matrix3D operator-(const Matrix3D &left, const Matrix3D &right);
Matrix3D operator*(const Matrix3D &left, const Matrix3D &right);
Matrix3D operator*(const Matrix3D &left, const float right);
Matrix3D operator*(const float left, const Matrix3D &right);
Vector3D operator*(const Matrix3D &left, const Vector3D &right);

std::ostream &operator<<(std::ostream &stream, const Matrix3D &matrix);

} // namespace Math

#endif // MATRIX3D_H
