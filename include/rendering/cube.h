#ifndef CUBE_H
#define CUBE_H

#include <stddef.h>

#include "abstractmesh.h"
#include "triangle.h"

namespace Rendering
{

class Cube : AbstractMesh
{
public:
	Cube(const double sideLength = 1);
	virtual ~Cube();
};

} // namespace Rendering

#endif // CUBE_H
