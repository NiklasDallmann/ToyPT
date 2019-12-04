#ifndef TILE_H
#define TILE_H

#include <stdint.h>

namespace ToyPT
{
namespace Rendering
{

struct Tile
{
	uint32_t	x0;
	uint32_t	y0;
	uint32_t	x1;
	uint32_t	y1;
};

}
}

#endif // TILE_H
