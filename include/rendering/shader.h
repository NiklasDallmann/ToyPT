#ifndef SHADER_H
#define SHADER_H

#include <cxxutility/definitions.h>
#include <math/vector4.h>
#include <cmath>

#include "material.h"

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
	
	HOST_DEVICE static inline Math::Vector4 diffuseLambert(const Math::Vector4 &diffuseColor)
	{
		return (1.0f / float(M_PI)) * diffuseColor;
	}
	
	HOST_DEVICE static inline Math::Vector4 specularCookTorrance(
		const Material &material,
		const Math::Vector4 &v,
		const Math::Vector4 &l,
		const Math::Vector4 &n)
	{
		const Math::Vector4	h			= (l + v).normalized();
		const float			nDotH		= n.dotProduct(h);
		const float			vDotH		= v.dotProduct(h);
		const float			a			= material.roughness;
		
		Math::Vector4		numerator;
							numerator	= _distribution_ggx(nDotH, n, a);
							numerator	*= _fresnel_schlick(vDotH, material.reflectance);
							numerator	*= _geometric_ggx(v, l, n, a * a);
		
		const float			denominator	= 4.0f * n.dotProduct(v) * n.dotProduct(l);
		
		return numerator / denominator;
	}
	
	HOST_DEVICE static inline Math::Vector4 schlickF0(const float indexOfRefraction)
	{
		float	f0	= indexOfRefraction - 1.0f;
				f0	/= indexOfRefraction + 1.0f;
		
		return Math::Vector4{f0 * f0};
	}
	
	HOST_DEVICE static inline Math::Vector4 encodeGamma(const Math::Vector4 &pixel, const float exponent)
	{
		Math::Vector4	returnValue;
		const float		inverseExponent = 1.0f / exponent;
		
#ifndef NVCC
		returnValue = {std::pow(pixel.x(), inverseExponent), std::pow(pixel.y(), inverseExponent), std::pow(pixel.z(), inverseExponent)};
#else
		returnValue = {powf(pixel.x(), inverseExponent), powf(pixel.y(), inverseExponent), powf(pixel.z(), inverseExponent)};
#endif
		
		return returnValue;
	}
	
private:
	HOST_DEVICE static inline Math::Vector4 _distribution_ggx(const float &nDotH, const Math::Vector4 &n, const float a)
	{
		const float	d	= nDotH * a;
		const float	k	= a / (1.0f - nDotH * nDotH + d * d);
		
		return k * k * (1.0 / float(M_PI));
	}
	
	HOST_DEVICE static inline float _geometric_ggx(const Math::Vector4 &v, const Math::Vector4 &l, const Math::Vector4 &n, const float a_2)
	{
		return _ggxPartial(v, n, a_2) * _ggxPartial(v, n, a_2);
	}
	
	HOST_DEVICE static inline float _ggxChi(const float x)
	{
		return (x > 0.0f) ? 1.0f : 0.0f;
	}
	
	HOST_DEVICE static inline float _ggxPartial(const Math::Vector4 &v, const Math::Vector4 &n, const float a_2)
	{
		const float nDotV		= n.dotProduct(v);
		const float	numerator	= 2.0f * nDotV;
		const float	radicand	= a_2 + (1.0f - a_2) * nDotV * nDotV;
		
#ifndef NVCC
		const float denominator	= nDotV + std::sqrt(radicand);
#else
		const float denominator	= nDotV + sqrtf(radicand);
#endif
		
		return numerator / denominator;
	}
	
	HOST_DEVICE static inline Math::Vector4 _fresnel_schlick(const float vDotH, const Math::Vector4 &f0)
	{
#ifndef NVCC
		const float f = std::pow(1.0 - vDotH, 5.0f);
#else
		const float f = powf(1.0 - vDotH, 5.0f);
#endif
		
		return f + f0 * (1.0f - f);
	}
};

} // namespace Rendering
} // namespace ToyPT

#endif // SHADER_H
