#include "cuda/cudaarray.h"
#include "cuda/cudatypes.h"
#include "rendering/framebuffer.h"
#include "math/algorithms.h"
#include "math/matrix4x4.h"
#include "math/vector4.h"
#include "rendering/ray.h"
#include "rendering/randomnumbergenerator.h"

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
//		auto &triangle = scene.triangleBuffer[triangleIndex];
		
//		printf("v0  := {%f, %f, %f}\n", triangle.v0.x(), triangle.v0.y(), triangle.v0.z());
//		printf("e01 := {%f, %f, %f}\n", triangle.e01.x(), triangle.e01.y(), triangle.e01.z());
//		printf("e02 := {%f, %f, %f}\n", triangle.e02.x(), triangle.e02.y(), triangle.e02.z());
//		printf("e12 := {%f, %f, %f}\n", triangle.e12.x(), triangle.e12.y(), triangle.e12.z());
		
		float t, u, v;
		bool intersected = intersect(ray, &dataPointer, t, u, v);
		
		if (intersected == true)
		{
			printf("intersection %f\n", t);
		}
		else
		{
			printf("no intersection %f\n", t);
		}
		
		if ((newDistance < distance) & (intersected == true))
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

__device__ Math::Vector4 createUniformHemisphere(const float r1, const float r2)
{
	float sinTheta	= sqrtf(1.0f - r1 * r1);
	float phi		= 2.0f * float(M_PI) * r2;
	float x			= sinTheta * cosf(phi);
	float z			= sinTheta * sinf(phi);
	
	return {x, r1, z};
}

__device__ Math::Vector4 randomDirection(const Math::Vector4 &normal, RandomNumberGenerator &rng, float &cosinusTheta)
{
	float ratio;
	
	Math::Vector4 Nt;
	Math::Vector4 Nb;
	
	createCoordinateSystem(normal, Nt, Nb);
	
	// Generate hemisphere
	constexpr float scalingFactor	= 1.0f / float(0xFFFFFFFF);
	cosinusTheta					= std::pow(rng.get(scalingFactor), 0.5f);
	ratio							= rng.get(scalingFactor);
	
	Math::Vector4 sample			= createUniformHemisphere(cosinusTheta, ratio);
	
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

__device__ Ray createCameraRay(const uint pixelX, const uint pixelY, const uint width, const uint height, const float fieldOfView, RandomNumberGenerator &rng)
{
	float fovRadians = fieldOfView / 180.0f * float(M_PI);
	float zCoordinate = -(width/(2.0f * tanf(fovRadians / 2.0f)));
	
	float offsetX, offsetY;
	constexpr float scalingFactor = 1.0f / float(0xFFFFFFFF);
	offsetX = rng.get(scalingFactor)  - 0.5f;
	offsetY = rng.get(scalingFactor) - 0.5f;
	
	float x = (pixelX + offsetX + 0.5f) - (width / 2.0f);
	float y = -(pixelY + offsetY + 0.5f) + (height / 2.0f);
	
	Math::Vector4 direction{x, y, zCoordinate};
	direction.normalize();
	
	return Ray{Math::Vector4{}, direction};
}

__global__ void castRay(const Cuda::Types::Tile tile, RandomNumberGenerator *rngs, const uint32_t width, const uint32_t height, const float fieldOfView,
						const Cuda::Types::Scene scene, const size_t maxBounces, const Math::Vector4 skyColor, Math::Vector4 *pixels)
{
	const uint pixelX		= blockIdx.x * blockDim.x + threadIdx.x;
	const uint pixelY		= blockIdx.y * blockDim.y + threadIdx.y;
	const uint pixelIndex	= pixelY * width + pixelX;
	
	// Return early if pixel is outside tile
	if ((pixelX >= tile.x1) | (pixelY >= tile.y1))
	{
		return;
	}
	
	RandomNumberGenerator &rng = rngs[pixelIndex];
	Ray ray = createCameraRay(pixelX, pixelY, width, height, fieldOfView, rng);
	
	Math::Vector4 returnValue = {0.0f, 0.0f, 0.0f};
	Math::Vector4 mask = {1.0f, 1.0f, 1.0f};
	
	Math::Vector4 currentDirection = ray.direction;
	Math::Vector4 currentOrigin = ray.origin;
	
	float cosinusTheta;
	
//	for (size_t currentBounce = 0; currentBounce < maxBounces; currentBounce++)
	{
		Math::Vector4 intersectionPoint;
		Types::IntersectionInfo objectIntersection;
		Math::Vector4 normal;
		float objectDistance = traceRay({currentOrigin, currentDirection}, scene, objectIntersection);
		
		intersectionPoint = currentOrigin + (objectDistance * currentDirection);
		
		if (objectIntersection.mesh != nullptr)
		{
			Material objectMaterial = scene.materialBuffer[objectIntersection.mesh->materialOffset];
			Math::Vector4 objectColor = objectMaterial.color;
			
			// Calculate normal
			const Cuda::Types::Triangle *dataPointer = scene.triangleBuffer + objectIntersection.triangleOffset;
			normal = interpolateNormal(intersectionPoint, dataPointer);
			
			// Calculate new origin and offset
//			currentOrigin = intersectionPoint + (Math::epsilon * normal);
			
//			// Global illumination
//			Math::Vector4 newDirection, reflectedDirection, diffuseDirection;
//			Math::Vector4 diffuse, specular;
			
//			diffuseDirection = randomDirection(normal, rng, cosinusTheta);
//			reflectedDirection = (currentDirection - 2.0f * currentDirection.dotProduct(normal) * normal).normalize();
			
//			newDirection = Math::lerp(diffuseDirection, reflectedDirection, objectMaterial.roughness);
			
//			specular = Math::Vector4{1.0f, 1.0f, 1.0f} * (1.0f - objectMaterial.roughness);
//			diffuse = Math::Vector4{1.0f, 1.0f, 1.0f} - specular;
			
//			currentDirection = newDirection;
			
//			returnValue += objectMaterial.emittance * mask;
//			mask *= (2.0f * objectColor * diffuse + specular) * cosinusTheta;
			returnValue = objectColor;
//			returnValue = {0.0f, 1.0f, 0.0f};
		}
		else
		{
//			returnValue = {1.0f, 0.0f, 0.0f};
			returnValue += skyColor * mask;
//			break;
		}
	}
	
	pixels[pixelIndex] = returnValue;
}

__host__ void cudaRender(
	FrameBuffer &frameBuffer, RandomNumberGenerator rng,
	const CudaArray<Types::Triangle> &triangleBuffer,
	const CudaArray<Types::Mesh> &meshBuffer,
	const CudaArray<Material> &materialBuffer,
	const uint32_t samples,
	const uint32_t maxBounces,
	const float fieldOfView,
	const Math::Vector4 &skyColor)
{
	const uint32_t pixelCount		= frameBuffer.width() * frameBuffer.height();
	const uint32_t threadsPerBlock	= 32;
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
	
	dim3 blockSize(threadsPerBlock, threadsPerBlock, 1);
	dim3 gridSize(gridSizeX, gridkSizeY, 1);
	
	printf("width=%u height=%u\n", frameBuffer.width(), frameBuffer.height());
	
	CudaArray<RandomNumberGenerator> rngs(pixelCount);
	
	for (uint32_t i = 0; i < rngs.size(); i++)
	{
		rngs[i] = rng.get();
	}
	
	printf("created rngs\n");
	
	CudaArray<Math::Vector4> gpuFrameBuffer(pixelCount);
	
	printf("allocated GPU framebuffer... launching kernel...\n");
	
	printf("gridSize=(%u, %u, %u) blockSize=(%u, %u, %u)\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
	
	castRay<<<gridSize, blockSize>>>(
		Types::Tile{0, 0, frameBuffer.width(), frameBuffer.height()},
		rngs.data(),
		frameBuffer.width(),
		frameBuffer.height(),
		fieldOfView,
		scene,
		maxBounces,
		skyColor,
		gpuFrameBuffer.data());
	
	cudaDeviceSynchronize();
	
	handleCudaError(cudaGetLastError());
	
	printf("finished kernel execution\n");
	
	// Copy frame buffer back
	frameBuffer = FrameBuffer::fromRawData(gpuFrameBuffer.data(), frameBuffer.width(), frameBuffer.height());
}

}
}
}
