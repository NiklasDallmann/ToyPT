#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

//#include "abstractrenderer.h"
#include "cuda/cudaarray.h"
#include "cuda/cudatypes.h"
#include "rendering/framebuffer.h"
#include "math/algorithms.h"
#include "math/matrix4x4.h"
#include "math/vector4.h"
#include "rendering/ray.h"
#include "rendering/randomnumbergenerator.h"
#include "rendering/shader.h"

namespace ToyPT
{
namespace Rendering
{
namespace Cuda
{

__host__ void handleCudaError(const cudaError error)
{
	if (error != cudaSuccess)
	{
		printf("CUDA Error: %s %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
	}
}

__device__ void print(const char *name, const Math::Vector4 &vector)
{
	printf("%s := {%f, %f, %f, %f}\n", name, vector.x(), vector.y(), vector.z(), vector.w());
}

__device__ bool intersect(const Rendering::Ray &ray, CudaArray<Cuda::Types::Triangle>::const_pointer *data, float &t, float &u, float &v)
{
	bool returnValue;
	float determinant, inverseDeterminant;
	
	Cuda::Types::Triangle triangle	= **data;
	Math::Vector4 v0o, pVector, qVector;
	
	// Sub
	v0o = ray.origin - triangle.v0;
	
	// Cross
	pVector = ray.direction.crossProduct(triangle.e02);
	qVector = v0o.crossProduct(triangle.e01);
	
	determinant = triangle.e01.dotProduct(pVector);
	inverseDeterminant = 1.0f / determinant;
	
	u = v0o.dotProduct(pVector) * inverseDeterminant;
	v = ray.direction.dotProduct(qVector) * inverseDeterminant;
	t = triangle.e02.dotProduct(qVector) * inverseDeterminant;
	(*data)++;
	
	// Conditions
	bool c0, c1, c2, c3, c4, c5;
	c0 = determinant < Math::epsilon;
	c1 = u < 0.0f;
	c2 = u > 1.0f;
	c3 = v < 0.0f;
	c4 = (u + v) > 1.0f;
	c5 = t > Math::epsilon;
	returnValue = !(c0 | c1 | c2 | c3 | c4) & c5;
	
	return returnValue;
}

__device__ float traceRay(const Rendering::Ray &ray, const Cuda::Types::Scene &scene, Cuda::Types::IntersectionInfo &intersection)
{
	float returnValue							= 0.0f;
	
	const Cuda::Types::Triangle *dataPointer	= scene.triangleBuffer;
	const Cuda::Types::Mesh *nearestMesh		= nullptr;
	uint32_t nearestTriangle					= 0xFFFFFFFF;
	float distance								= 1E7f;
	float newDistance							= distance;
//	float u = 0;
//	float v = 0;
	
	// Intersect triangles
	for (uint32_t triangleIndex = 0; triangleIndex < scene.triangleCount; triangleIndex++)
	{
		float u, v;
		bool intersected = intersect(ray, &dataPointer, newDistance, u, v);
		
		if ((newDistance < distance) & intersected)
		{
			nearestMesh		= &scene.meshBuffer[scene.triangleBuffer[triangleIndex].meshIndex];
			distance		= newDistance;
			nearestTriangle	= triangleIndex;
		}
	}
	
	returnValue					= distance;
	intersection.mesh			= nearestMesh;
	intersection.triangleOffset	= nearestTriangle;
//	intersection.u = u;
//	intersection.v = v;
	
	return returnValue;
}

__device__ void createCoordinateSystem(const Math::Vector4 &normal, Math::Vector4 &tangentNormal, Math::Vector4 &binormal)
{
	const Math::Vector4 a	= Math::Vector4{normal.z(), 0.0f, -normal.x()};
	const Math::Vector4 b	= Math::Vector4{0.0f, -normal.z(), normal.y()};
	float t					= fabsf(normal.x()) > fabsf(normal.y());
	
	tangentNormal			= Math::lerp(a, b, t).normalize();
	
	binormal				= normal.crossProduct(tangentNormal);
}

__device__ Math::Vector4 sampleUniform(const float r1, const float r2)
{
	float sinTheta	= sqrtf(1.0f - r1 * r1);
	float phi		= 2.0f * float(M_PI) * r2;
	float x			= sinTheta * cosf(phi);
	float z			= sinTheta * sinf(phi);
	
	return {x, r1, z};
}

__device__ Math::Vector4 sampleImportant(const float r1, const float r2, const float a)
{
	float theta		= acosf(sqrtf((1.0f - r1) / ((a * a - 1.0f) * r1 + 1.0f)));
	float phi		= 2.0f * float(M_PI) * r2;
	float x			= cosf(phi) * sinf(theta);
	float y			= cosf(theta);
	float z			= sinf(phi) * sinf(theta);
	
	return {x, y, z};
}

__device__ Math::Vector4 randomDirection(const Math::Vector4 &normal, curandState &rng, float &cosinusTheta, const float roughness)
{
//	float ratio;
	
	Math::Vector4 Nt;
	Math::Vector4 Nb;
	
	createCoordinateSystem(normal, Nt, Nb);
	
	// Generate hemisphere
//	cosinusTheta			= Math::pow(curand_uniform(&rng), 0.5f);
//	ratio					= curand_uniform(&rng);
	
//	Math::Vector4 sample	= sampleUniform(cosinusTheta, ratio);
	Math::Vector4 sample	= sampleImportant(curand_uniform(&rng), curand_uniform(&rng), roughness);
	
	Math::Matrix4x4 localToWorldMatrix{
		{Nb.x(), normal.x(), Nt.x()},
		{Nb.y(), normal.y(), Nt.y()},
		{Nb.z(), normal.z(), Nt.z()}
	};
	
	return (localToWorldMatrix * sample).normalize();
}

__device__ Math::Vector4 interpolateNormal(const Math::Vector4 &intersectionPoint, const Types::Triangle *data)
{
	Math::Vector4 returnValue, p, n0, n1, n2, n01, n02, v0, v1, v2, e01, e02, v12, v0p, v1p, v2p, vab, v2ab;
	
	v0	= data->v0;
	e01	= data->e01;
	e02	= data->e02;
	v1	= e01 + v0;
	v2	= e02 + v0;
	n0	= data->n0;
	n1	= data->n1;
	n2	= data->n2;
	
	data++;
	
	p	= intersectionPoint;
	v12	= v2 - v1;
	v0p	= p - v0;
	v1p	= p - v1;
	v2p	= p - v2;
	
	float a, denominator;
	
	denominator = (e01.x() * v2p.y() - v2p.x() * e01.y()) + Math::epsilon;
	a			= (-(v0.x() * v2p.y() - v2p.x() * v0.y() + v2p.x() * v2.y() - v2.x() * v2p.y())) / denominator;
//	b			= (e01.x() * v0.y() - e01.x() * v2.y() - v0.x() * e01.y() + v2.x() * e01.y()) / denominator;
	
	vab		= v0 + a * e01;
	
	n01		= Math::lerp(n1, n0, a).normalize();
	v2ab	= vab - v2;
	
	returnValue = Math::lerp(n01, n2, (v2p.magnitude() / v2ab.magnitude())).normalize();
	
	return returnValue;
}

__device__ Ray createCameraRay(const uint pixelX, const uint pixelY, const uint width, const uint height, const float fieldOfView, curandState &rng)
{
	float fovRadians	= fieldOfView / 180.0f * float(M_PI);
	float zCoordinate	= -(width/(2.0f * tanf(fovRadians / 2.0f)));
	
	float offsetX, offsetY;
	offsetX = curand_uniform(&rng)  - 0.5f;
	offsetY = curand_uniform(&rng) - 0.5f;
	
	float x = (pixelX + offsetX + 0.5f) - (width / 2.0f);
	float y = -(pixelY + offsetY + 0.5f) + (height / 2.0f);
	
	Math::Vector4 direction{x, y, zCoordinate};
	direction.normalize();
	
	return Ray{Math::Vector4{}, direction};
}

__global__ void castRay(const Cuda::Types::Tile tile, curandState *rngs, const uint32_t width, const uint32_t height, const float fieldOfView,
						const Cuda::Types::Scene scene, const size_t maxBounces, const Math::Vector4 skyColor, Math::Vector4 *pixels)
{
	const uint pixelX				= blockIdx.x * blockDim.x + threadIdx.x;
	const uint pixelY				= blockIdx.y * blockDim.y + threadIdx.y;
	const uint pixelIndex			= pixelY * width + pixelX;
	
	// Return early if pixel is outside tile
	if ((pixelX >= tile.x1) | (pixelY >= tile.y1))
	{
		return;
	}
	
	curandState rng					= rngs[pixelIndex];
	Ray ray							= createCameraRay(pixelX, pixelY, width, height, fieldOfView, rng);
	
	Math::Vector4 returnValue		= pixels[pixelIndex];
	Math::Vector4 mask				= {1.0f, 1.0f, 1.0f};
	
	Math::Vector4 currentDirection	= ray.direction;
	Math::Vector4 currentOrigin		= ray.origin;
	
	float cosinusTheta;
	
	for (size_t currentBounce = 0; currentBounce < maxBounces; currentBounce++)
	{
		Math::Vector4 intersectionPoint;
		Types::IntersectionInfo objectIntersection;
		Math::Vector4 normal;
		float objectDistance	= traceRay({currentOrigin, currentDirection}, scene, objectIntersection);
		
		intersectionPoint		= currentOrigin + (objectDistance * currentDirection);
		
		if (objectIntersection.mesh != nullptr)
		{
			const Cuda::Types::Triangle *dataPointer;
			Math::Vector4 objectColor, newDirection, diffuse, specular, kd, ks;
			Material objectMaterial		= scene.materialBuffer[objectIntersection.mesh->materialOffset];
			
			objectColor					= objectMaterial.color;
			
			// Calculate normal
			dataPointer					= scene.triangleBuffer + objectIntersection.triangleOffset;
			normal						= interpolateNormal(intersectionPoint, dataPointer);
			
			// Calculate new origin and offset
			currentOrigin				= intersectionPoint + (Math::epsilon * normal);
			
			// Global illumination
			newDirection				= randomDirection(normal, rng, cosinusTheta, objectMaterial.roughness * objectMaterial.roughness);
			cosinusTheta				= normal.dotProduct(newDirection);
			
			diffuse						= objectColor * Shader::diffuseLambert();
			specular					= Shader::specularCookTorrance(objectMaterial, -currentDirection, newDirection, normal, ks);
			kd							= (Math::Vector4{1.0f} - ks) * (1.0f - objectMaterial.metallic);
			
			currentDirection			= newDirection;
			
			returnValue					+= objectMaterial.emittance * objectColor * mask;
			mask						*= ((kd * diffuse) + specular) * cosinusTheta;
		}
		else
		{
			returnValue					+= skyColor * mask;
			break;
		}
	}
	
	rngs[pixelIndex]	= rng;
	pixels[pixelIndex]	= returnValue;
}

__global__ void setupRngs(curandState *rngs, const uint32_t seed, const Cuda::Types::Tile tile, const uint32_t width)
{
	const uint pixelX		= blockIdx.x * blockDim.x + threadIdx.x;
	const uint pixelY		= blockIdx.y * blockDim.y + threadIdx.y;
	const uint pixelIndex	= pixelY * width + pixelX;
	
	// Return early if pixel is outside tile
	if ((pixelX >= tile.x1) | (pixelY >= tile.y1))
	{
		return;
	}
	
	curand_init(seed, pixelIndex, 0, &rngs[pixelIndex]);
}

__global__ void initializePixels(Math::Vector4 *pixels, const Cuda::Types::Tile tile, const uint32_t width)
{
	const uint pixelX		= blockIdx.x * blockDim.x + threadIdx.x;
	const uint pixelY		= blockIdx.y * blockDim.y + threadIdx.y;
	const uint pixelIndex	= pixelY * width + pixelX;
	
	// Return early if pixel is outside tile
	if ((pixelX >= tile.x1) | (pixelY >= tile.y1))
	{
		return;
	}
	
	pixels[pixelIndex] = {};
}

__global__ void finalizePixels(Math::Vector4 *pixels, const Cuda::Types::Tile tile, const uint32_t width, const uint32_t samples)
{
	const uint pixelX		= blockIdx.x * blockDim.x + threadIdx.x;
	const uint pixelY		= blockIdx.y * blockDim.y + threadIdx.y;
	const uint pixelIndex	= pixelY * width + pixelX;
	
	// Return early if pixel is outside tile
	if ((pixelX >= tile.x1) | (pixelY >= tile.y1))
	{
		return;
	}
	
	pixels[pixelIndex] /= float(samples);
}

__global__ void encodeGamma(Math::Vector4 *pixels, const Cuda::Types::Tile tile, const uint32_t width)
{
	const uint pixelX		= blockIdx.x * blockDim.x + threadIdx.x;
	const uint pixelY		= blockIdx.y * blockDim.y + threadIdx.y;
	const uint pixelIndex	= pixelY * width + pixelX;
	
	// Return early if pixel is outside tile
	if ((pixelX >= tile.x1) | (pixelY >= tile.y1))
	{
		return;
	}
	
	pixels[pixelIndex] = Shader::encodeGamma(pixels[pixelIndex], 2.2f);
}

__host__ void cudaRender(
	FrameBuffer							&frameBuffer,
	RandomNumberGenerator				rng,
	const CudaArray<Types::Triangle>	&triangleBuffer,
	const CudaArray<Types::Mesh>		&meshBuffer,
	const CudaArray<Material>			&materialBuffer,
//	const AbstractRenderer::CallBack	&callback,
	const bool							&abort,
	const uint32_t						samples,
	const uint32_t						maxBounces,
	const uint32_t						tileSize,
	const float							fieldOfView,
	const Math::Vector4					&skyColor)
{
	const uint32_t pixelCount		= frameBuffer.width() * frameBuffer.height();
	const uint32_t threadsPerBlock	= 16;
	const uint32_t gridSizeX		= (frameBuffer.width() / threadsPerBlock) + 1;
	const uint32_t gridkSizeY		= (frameBuffer.height() / threadsPerBlock) + 1;
	
	Types::Scene scene{
		triangleBuffer.data(),
		triangleBuffer.size(),
		meshBuffer.data(),
		meshBuffer.size(),
		materialBuffer.data(),
		materialBuffer.size()
	};
	
	Types::Tile tile{0, 0, frameBuffer.width(), frameBuffer.height()};
	
	dim3 blockSize(threadsPerBlock, threadsPerBlock, 1);
	dim3 gridSize(gridSizeX, gridkSizeY, 1);
	
	CudaArray<curandState> rngBuffer(pixelCount);
	CudaArray<Math::Vector4> gpuFrameBuffer(pixelCount);
	
	setupRngs<<<gridSize, blockSize>>>(
		rngBuffer.data(),
		rng.get(),
		Types::Tile{0, 0, frameBuffer.width(), frameBuffer.height()},
		frameBuffer.width());
	cudaDeviceSynchronize();
	handleCudaError(cudaGetLastError());
	
	initializePixels<<<gridSize, blockSize>>>(
		gpuFrameBuffer.data(),
		tile,
		frameBuffer.width());
	cudaDeviceSynchronize();
	handleCudaError(cudaGetLastError());
	
//	const uint32_t tilesVertical	= frameBuffer.height() / tileSize + ((frameBuffer.height() % tileSize) > 0);
//	const uint32_t tilesHorizontal	= frameBuffer.width() / tileSize + ((frameBuffer.width() % tileSize) > 0);
	
//	for (uint32_t tileVertical = 0; tileVertical < tilesVertical; tileVertical++)
//	{
//		for (uint32_t tileHorizontal = 0; tileHorizontal < tilesHorizontal; tileHorizontal++)
//		{
//			uint32_t startVertical		= tileSize * tileVertical;
//			uint32_t startHorizontal	= tileSize * tileHorizontal;
//			uint32_t endVertical		= std::min(startVertical + tileSize, frameBuffer.height());
//			uint32_t endHorizontal		= std::min(startHorizontal + tileSize, frameBuffer.width());
			
//			tile = {startHorizontal, startVertical, endHorizontal, endVertical};
			
			for (uint32_t sample = 0; sample < samples; sample++)
			{
				castRay<<<gridSize, blockSize>>>(
					tile,
					rngBuffer.data(),
					frameBuffer.width(),
					frameBuffer.height(),
					fieldOfView,
					scene,
					maxBounces,
					skyColor,
					gpuFrameBuffer.data());
				cudaDeviceSynchronize();
				handleCudaError(cudaGetLastError());
			}
			
//			if (!abort)
//			{
//				callBack(startHorizontal, startVertical, endHorizontal, endVertical);
//			}
//		}
//	}
	
	finalizePixels<<<gridSize, blockSize>>>(
		gpuFrameBuffer.data(),
		Types::Tile{0, 0, frameBuffer.width(), frameBuffer.height()},
		frameBuffer.width(),
		samples);
	cudaDeviceSynchronize();
	handleCudaError(cudaGetLastError());
	
	encodeGamma<<<gridSize, blockSize>>>(
		gpuFrameBuffer.data(),
		Types::Tile{0, 0, frameBuffer.width(), frameBuffer.height()},
		frameBuffer.width());
	cudaDeviceSynchronize();
	handleCudaError(cudaGetLastError());
	
	// Copy frame buffer back
	frameBuffer = FrameBuffer::fromRawData(gpuFrameBuffer.data(), frameBuffer.width(), frameBuffer.height());
}

}
}
}
