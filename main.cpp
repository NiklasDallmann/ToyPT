#include <iostream>

#include <matrix3d.h>
#include <vector3d.h>

int main()
{
	std::cout << "Path Tracer" << std::endl;
	
	Math::Matrix3D left = {
		{1, 2, 3},
		{4, 15, 6},
		{7, 8, 9}
	};
	
	Math::Matrix3D right = {
		{1, 1, 0},
		{0, 1, 0},
		{0, 0, 1}
	};
	
	std::cout << (left * right) << std::endl;
	std::cout << left.determinant() << std::endl;
	std::cout << left.inverted() << std::endl;
	
	return 0;
}
