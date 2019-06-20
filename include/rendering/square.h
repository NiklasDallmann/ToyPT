#ifndef SQUARE_H
#define SQUARE_H

#include "abstractmesh.h"

namespace Rendering
{

class Square : public AbstractMesh
{
public:
	Square(const size_t sideLength, const Material &material);
};

}

#endif // SQUARE_H
