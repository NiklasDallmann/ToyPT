#include <stdint.h>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>

#include <cxxutility/debugstream.h>

#include "cuda/cudarenderer.h"
#include "cuda/cudatypes.h"
#include "randomnumbergenerator.h"
#include "geometrycontainer.h"

namespace ToyPT::Rendering::Cuda
{

extern void cudaRender(FrameBuffer &frameBuffer, RandomNumberGenerator rng,
					 const CudaArray<Cuda::Types::Triangle> &triangleBuffer,
					 const CudaArray<Cuda::Types::Mesh> &meshBuffer,
					 const CudaArray<Material> &materialBuffer,
					 const uint32_t samples,
					 const uint32_t maxBounces,
					 const float fieldOfView,
					 const Math::Vector4 &skyColor);

void CudaRenderer::render(FrameBuffer &frameBuffer, const Obj::GeometryContainer &geometry, const Obj::GeometryContainer &lights,
						  const AbstractRenderer::CallBack &callBack, const bool &abort, const float fieldOfView, const uint32_t samples,
						  const uint32_t bounces, const uint32_t tileSize, const Math::Vector4 &skyColor)
{
	CudaArray<Cuda::Types::Triangle> triangleBuffer;
	CudaArray<Cuda::Types::Mesh> meshBuffer;
	CudaArray<Material> materialBuffer;
	
	this->_geometryToBuffer(geometry, triangleBuffer, meshBuffer, materialBuffer);
	
	cxxtrace << "starting render";
	
	std::random_device device;
	RandomNumberGenerator rng{device()};
	
	cudaRender(frameBuffer, rng, triangleBuffer, meshBuffer, materialBuffer, samples, bounces, fieldOfView, skyColor);
	
	cxxtrace << "finished render";
}

void CudaRenderer::_geometryToBuffer(const Obj::GeometryContainer &geometry, CudaArray<Cuda::Types::Triangle> &triangleBuffer,
									 CudaArray<Cuda::Types::Mesh> &meshBuffer, CudaArray<Material> &materialBuffer)
{
	std::vector<Cuda::Types::Triangle> triangles;
	std::vector<Cuda::Types::Mesh> meshes;
	
	for (uint32_t meshIndex = 0; meshIndex < geometry.meshBuffer.size(); meshIndex++)
	{
		const Obj::Mesh &objMesh = geometry.meshBuffer[meshIndex];
		Cuda::Types::Mesh mesh;
		mesh.triangleOffset = triangleBuffer.size();
		mesh.triangleCount = objMesh.triangleCount;
		mesh.materialOffset = objMesh.materialOffset;
		
		for (uint32_t triangleIndex = 0; triangleIndex < objMesh.triangleCount; triangleIndex++)
		{
			const Obj::Triangle &triangle = geometry.triangleBuffer[objMesh.triangleOffset + triangleIndex];
			
			Math::Vector4 v0, v1, v2, e01, e02, e12, n0, n1, n2;
			
			v0 = geometry.vertexBuffer[triangle.vertices[0]];
			v1 = geometry.vertexBuffer[triangle.vertices[1]];
			v2 = geometry.vertexBuffer[triangle.vertices[2]];
			e01 = v1 - v0;
			e02 = v2 - v0;
			e12 = v2 - v1;
			
			n0 = geometry.normalBuffer[triangle.normals[0]];
			n1 = geometry.normalBuffer[triangle.normals[1]];
			n2 = geometry.normalBuffer[triangle.normals[2]];
			
			triangles.push_back(Cuda::Types::Triangle{v0, e01, e02, e12, n0, n1, n2, meshIndex});
		}
		
		meshes.push_back(mesh);
	}
	
	triangleBuffer = CudaArray<Cuda::Types::Triangle>(CudaArray<Cuda::Types::Triangle>::size_type(triangles.size()));
	meshBuffer = CudaArray<Cuda::Types::Mesh>(CudaArray<Cuda::Types::Mesh>::size_type(meshes.size()));
	materialBuffer = CudaArray<Material>(CudaArray<Material>::size_type(geometry.materialBuffer.size()));
	
	for (CudaArray<Cuda::Types::Triangle>::size_type i = 0; i < triangles.size(); i++)
	{
		triangleBuffer[i] = triangles[i];
	}
	
	for (CudaArray<Cuda::Types::Mesh>::size_type i = 0; i < meshes.size(); i++)
	{
		meshBuffer[i] = meshes[i];
	}
	
	for (CudaArray<Material>::size_type i = 0; i < geometry.materialBuffer.size(); i++)
	{
		materialBuffer[i] = geometry.materialBuffer[i];
	}
}

}
