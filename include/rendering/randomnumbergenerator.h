#ifndef RANDOMNUMBERGENERATOR_H
#define RANDOMNUMBERGENERATOR_H

#include <stdint.h>
#include <cxxutility/definitions.h>

namespace ToyPT
{
namespace Rendering
{

class RandomNumberGenerator
{
public:
	HOST_DEVICE RandomNumberGenerator(const uint32_t seed) :
		_state(seed)
	{
	}
	
	HOST_DEVICE uint32_t get()
	{
		uint32_t x = this->_state;
		
		x = (x ^ 61) ^ (x >> 16);
		x *= 9;
		x = x ^ (x >> 4);
		x *= 0x27d4eb2d;
		x = x ^ (x >> 15);
		
//		x ^= (x << 13);
//		x ^= (x >> 17);
//		x ^= (x << 5);
		
		this->_state = x;
		
		return x;
	}
	
	HOST_DEVICE float get(const float scale)
	{
		return float(this->get()) * scale;
	}
	
private:
	uint32_t _state;
};

} // namespace Rendering
} // namespace ToyPT

#endif // RANDOMNUMBERGENERATOR_H
