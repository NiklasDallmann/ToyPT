#ifndef RENDERSETTINGS_H
#define RENDERSETTINGS_H

#include <stdint.h>

#include <math/vector4.h>

namespace ToyPT
{
namespace Rendering
{

struct RenderSettings
{
	uint32_t		width		= 400u;
	uint32_t		height		= 200u;
	float			fieldOfView	= 70.0f;
	uint32_t		samples		= 32u;
	uint32_t		bounces		= 4u;
	uint32_t		tileSize	= 32u;
	Math::Vector4	skyColor	= {};
};

}
}

#endif // RENDERSETTINGS_H
