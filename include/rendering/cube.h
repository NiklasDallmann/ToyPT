#ifndef CUBE_H
#define CUBE_H

#include <stddef.h>

#include "mesh.h"
#include "triangle.h"

namespace Rendering
{

class Cube : Mesh
{
public:
	Cube(const double sideLength = 1);
	virtual ~Cube();
};

} // namespace Rendering

#endif // CUBE_H
