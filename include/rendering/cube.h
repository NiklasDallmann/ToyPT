#ifndef CUBE_H
#define CUBE_H

#include <stddef.h>

#include "abstractmesh.h"
#include "material.h"
#include "triangle.h"

namespace Rendering
{

class Cube : public AbstractMesh
{
public:
	Cube(const float sideLength = 1, const Material &material = {});
};

} // namespace Rendering

#endif // CUBE_H
