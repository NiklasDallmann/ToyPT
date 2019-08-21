#ifndef RANDOMNUMBERGENERATOR_H
#define RANDOMNUMBERGENERATOR_H

#include <stdint.h>

namespace Rendering
{

class RandomNumberGenerator
{
public:
	RandomNumberGenerator(const uint32_t seed) :
		_state(seed)
	{
	}
	
	uint32_t get()
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
	
	float get(const float scale)
	{
		return float(this->get()) * scale;
	}
	
private:
	uint32_t _state;
};

} // namespace Rendering

#endif // RANDOMNUMBERGENERATOR_H
