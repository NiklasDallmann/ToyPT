#ifndef CUBE_H
#define CUBE_H

#include <stddef.h>

#include "mesh.h"
#include "material.h"
#include "triangle.h"

namespace Rendering
{

class Cube : public Mesh
{
public:
	Cube(const float sideLength = 1, const Material &material = {});
};

} // namespace Rendering

#endif // CUBE_H
