#include "vector3d.h"

#include <cmath>
#include <sstream>

namespace Math
{

Vector3D::Vector3D(const double x, const double y, const double z, const double w) :
	_coordinates({x, y, z, w})
{
}

double Vector3D::x() const
{
	return this->_coordinates[0];
}

void Vector3D::setX(const double x)
{
	this->_coordinates[0] = x;
}

double Vector3D::y() const
{
	return this->_coordinates[1];
}

void Vector3D::setY(const double y)
{
	this->_coordinates[1] = y;
}

double Vector3D::z() const
{
	return this->_coordinates[2];
}

void Vector3D::setZ(const double z)
{
	this->_coordinates[2] = z;
}

double Vector3D::magnitude() const
{
	return std::pow((std::pow(this->_coordinates[0], 2) + std::pow(this->_coordinates[1], 2) + std::pow(this->_coordinates[2], 2)), 0.5);
}

Vector3D &Vector3D::normalize()
{
	double magnitude = this->magnitude();
	
	this->_coordinates[0] /= magnitude;
	this->_coordinates[1] /= magnitude;
	this->_coordinates[2] /= magnitude;
	
	return *this;
}

Vector3D Vector3D::normalized() const
{
	Vector3D returnValue = *this;
	
	returnValue.normalize();
	
	return returnValue;
}

Vector3D Vector3D::crossProduct(const Vector3D &other) const
{
	Vector3D returnValue;
	
	returnValue[0] = (*this)[1] * other[2] - (*this)[2] * other[1];
	returnValue[1] = (*this)[2] * other[0] - (*this)[0] * other[2];
	returnValue[2] = (*this)[0] * other[1] - (*this)[1] * other[0];
	
	return returnValue;
}

Vector3D Vector3D::coordinateProduct(const Vector3D &other) const
{
	Vector3D returnValue;
	
	returnValue[0] = (*this)[0] * other[0];
	returnValue[1] = (*this)[1] * other[1];
	returnValue[2] = (*this)[2] * other[2];
	
	return returnValue;
}

double Vector3D::cos(const Vector3D &left, const Vector3D &right)
{
	double returnValue = 0;
	
	returnValue = ((left * right) / (left.magnitude() * right.magnitude()));
	
	return returnValue;
}

Vector3D &Vector3D::operator+=(const Vector3D &other)
{
	this->_coordinates[0] += other._coordinates[0];
	this->_coordinates[1] += other._coordinates[1];
	this->_coordinates[2] += other._coordinates[2];
	
	return *this;
}

Vector3D &Vector3D::operator-=(const Vector3D &other)
{
	this->_coordinates[0] -= other._coordinates[0];
	this->_coordinates[1] -= other._coordinates[1];
	this->_coordinates[2] -= other._coordinates[2];
	
	return *this;
}

Vector3D &Vector3D::operator*=(const double scalar)
{
	this->_coordinates[0] *= scalar;
	this->_coordinates[1] *= scalar;
	this->_coordinates[2] *= scalar;
	
	return *this;
}

Vector3D &Vector3D::operator/=(const double scalar)
{
	this->_coordinates[0] /= scalar;
	this->_coordinates[1] /= scalar;
	this->_coordinates[2] /= scalar;
	
	return *this;
}

Vector3D Vector3D::operator-() const
{
	return *this * (-1.0);
}

double &Vector3D::operator[](const size_t index)
{
	return this->_coordinates[index];
}

double Vector3D::operator[](const size_t index) const
{
	return this->_coordinates[index];
}

bool Vector3D::operator==(const Vector3D &other)
{
	return (this->_coordinates == other._coordinates);
}

bool Vector3D::operator!=(const Vector3D &other)
{
	return !(*this == other);
}

Vector3D operator+(const Vector3D &left, const Vector3D &right)
{
	Vector3D returnValue = left;
	
	returnValue += right;
	
	return returnValue;
}

Vector3D operator-(const Vector3D &left, const Vector3D &right)
{
	Vector3D returnValue = left;
	
	returnValue -= right;
	
	return returnValue;
}

double operator*(const Vector3D &left, const Vector3D &right)
{
	double returnValue = 0;
	
	returnValue += (left._coordinates[0] * right._coordinates[0]);
	returnValue += (left._coordinates[1] * right._coordinates[1]);
	returnValue += (left._coordinates[2] * right._coordinates[2]);
	
	return returnValue;
}

Vector3D operator*(const Vector3D &left, const double right)
{
	Vector3D returnValue = left;
	
	returnValue *= right;
	
	return returnValue;
}

Vector3D operator*(const double left, const Vector3D right)
{
	return right * left;
}

Vector3D operator/(const Vector3D &left, const double right)
{
	Vector3D returnValue = left;
	
	returnValue /= right;
	
	return returnValue;
}

Vector3D operator/(const double left, const Vector3D right)
{
	return right / left;
}

std::ostream &operator<<(std::ostream &stream, const Vector3D &vector)
{
	std::stringstream stringStream;
	
	stringStream << "{\"x\": " << vector._coordinates[0] << ", \"y\": " << vector._coordinates[1] << ", \"z\": " << vector._coordinates[2] << "}";
	
	stream << stringStream.str();
	
	return stream;
}

}
