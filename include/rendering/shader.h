#ifndef SHADER_H
#define SHADER_H

#include <cmath>
#include <cxxutility/definitions.h>
#include <math/algorithms.h>
#include <math/vector4.h>

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
	
	HOST_DEVICE static inline Math::Vector4 diffuseLambert()
	{
		return (1.0f / float(M_PI));
	}
	
	HOST_DEVICE static inline Math::Vector4 specularCookTorrance(
		const Material &material,
		const Math::Vector4 &v,
		const Math::Vector4 &l,
		const Math::Vector4 &n)
	{
		const Math::Vector4	h			= (l + v).normalized();
		const float			nDotH		= Math::saturate(n.dotProduct(h));
		const float			vDotH		= Math::saturate(v.dotProduct(h));
		const float			a			= Math::pow(material.roughness, 2.0f);
		
		Math::Vector4		numerator;
							numerator	= _distribution_ggx(nDotH, a);
							numerator	*= _fresnel_schlick(vDotH, material.reflectance);
							numerator	*= _geometric_ggx(v, n, a * a);
		
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
		
		returnValue = {Math::pow(pixel.x(), inverseExponent), Math::pow(pixel.y(), inverseExponent), Math::pow(pixel.z(), inverseExponent)};
		
		return returnValue;
	}
	
private:
	HOST_DEVICE static inline Math::Vector4 _distribution_ggx(const float &nDotH, const float a)
	{
		const float	d	= nDotH * a;
		const float	k	= a / (1.0f - nDotH * nDotH + d * d);
		
		return k * k * (1.0f / float(M_PI));
	}
	
	HOST_DEVICE static inline float _geometric_ggx(const Math::Vector4 &v, const Math::Vector4 &n, const float a_2)
	{
		return _ggxPartial(v, n, a_2) * _ggxPartial(v, n, a_2);
	}
	
	HOST_DEVICE static inline float _ggxChi(const float x)
	{
		return (x > 0.0f) ? 1.0f : 0.0f;
	}
	
	HOST_DEVICE static inline float _ggxPartial(const Math::Vector4 &v, const Math::Vector4 &n, const float a_2)
	{
		const float nDotV		= Math::saturate(n.dotProduct(v));
		const float	numerator	= 2.0f * nDotV;
		const float	radicand	= a_2 + (1.0f - a_2) * nDotV * nDotV;
		
		const float denominator	= nDotV + Math::sqrt(radicand);
		
		return numerator / denominator;
	}
	
	HOST_DEVICE static inline Math::Vector4 _fresnel_schlick(const float vDotH, const Math::Vector4 &f0)
	{
		const float f = Math::pow(1.0f - vDotH, 5.0f);
		
		return f + f0 * (1.0f - f);
	}
};

} // namespace Rendering
} // namespace ToyPT

#endif // SHADER_H
