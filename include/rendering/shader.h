#ifndef SHADER_H
#define SHADER_H

#include <math/vector4.h>

#include "material.h"
#include "randomnumbergenerator.h"
#include "storage.h"

namespace ToyPT
{
namespace Rendering
{

class Shader
{
public:
	Shader() = delete;
	Shader(const Shader &other) = delete;
	Shader(Shader &&other) = delete;
	~Shader() = delete;
	
//	static inline void createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal);
//	static inline Math::Vector4 createUniformHemisphere(const float r1, const float r2);
//	static inline Math::Vector4 randomDirection(const Math::Vector4 &normal, RandomNumberGenerator &rng, float &cosinusTheta);
//	static inline Math::Vector4 interpolateNormal(const Math::Vector4 &intersectionPoint, Storage::PrecomputedTrianglePointer &data);
	
	static inline Math::Vector4 shade(
		const Material &material,
		const Math::Vector4 &n,
		const Math::Vector4 &l,
		const Math::Vector4 &v,
		const float cosinusTheta)
	{
		Math::Vector4 returnValue;
		
		
		
		return returnValue;
	}
	
private:
	static inline float _ggxChi(const float x)
	{
		float returnValue = 0.0f;
		
		
		
		return returnValue;
	}
	
	static inline float _ggxPartial(const Math::Vector4 &v, const Math::Vector4 &h, const Math::Vector4 &n, const float a_2)
	{
		float returnValue = 0.0f;
		
		
		
		return returnValue;
	}
};

} // namespace Rendering
} // namespace ToyPT

#endif // SHADER_H
