#ifndef SQUARE_H
#define SQUARE_H

#include "mesh.h"

namespace Rendering
{

class Square : public Mesh
{
public:
	Square(const size_t sideLength, const Material &material);
};

}

#endif // SQUARE_H
