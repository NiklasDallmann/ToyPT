#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <stack>
#include <stdint.h>
#include <vector>

#include <cxxutility/debugstream.h>

#include "cuda/cudarenderer.h"
#include "cuda/cudatypes.h"
#include "kdtreebuilder.h"
#include "randomnumbergenerator.h"
#include "geometrycontainer.h"

namespace ToyPT::Rendering::Cuda
{

extern void cudaRender(
		FrameBuffer							&frameBuffer,
		const RenderSettings				&settings,
		RandomNumberGenerator				rng,
		const CudaArray<Types::Triangle>	&triangleBuffer,
		const CudaArray<Types::Mesh>		&meshBuffer,
		const CudaArray<Material>			&materialBuffer,
		const bool							&abort);

void CudaRenderer::render(
		FrameBuffer						&frameBuffer,
		const RenderSettings			&settings,
		const Obj::GeometryContainer	&geometry, 
		const CallBack					&callBack,
		const bool						&abort)
{
	CudaArray<Cuda::Types::Node> nodeBuffer;
	CudaArray<Cuda::Types::Triangle> triangleBuffer;
	CudaArray<Cuda::Types::Mesh> meshBuffer;
	CudaArray<Material> materialBuffer;
	
	this->_geometryToBuffer(geometry, triangleBuffer, meshBuffer, materialBuffer);
//	this->_buildKdTree(geometry, nodeBuffer, triangleBuffer, meshBuffer, materialBuffer);
	
	cxxtrace << "starting render";
	
	std::random_device device;
	RandomNumberGenerator rng{device()};
	
	cudaRender(frameBuffer, settings, rng, triangleBuffer, meshBuffer, materialBuffer, abort);
	
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
			
			v0	= geometry.vertexBuffer[triangle.vertices[0]];
			v1	= geometry.vertexBuffer[triangle.vertices[1]];
			v2	= geometry.vertexBuffer[triangle.vertices[2]];
			e01	= v1 - v0;
			e02	= v2 - v0;
			e12	= v2 - v1;
			
			n0	= geometry.normalBuffer[triangle.normals[0]];
			n1	= geometry.normalBuffer[triangle.normals[1]];
			n2	= geometry.normalBuffer[triangle.normals[2]];
			
			triangles.push_back(Cuda::Types::Triangle{v0, e01, e02, e12, n0, n1, n2, meshIndex});
		}
		
		meshes.push_back(mesh);
	}
	
	triangleBuffer	= CudaArray<Cuda::Types::Triangle>(CudaArray<Cuda::Types::Triangle>::size_type(triangles.size()));
	meshBuffer		= CudaArray<Cuda::Types::Mesh>(CudaArray<Cuda::Types::Mesh>::size_type(meshes.size()));
	materialBuffer	= CudaArray<Material>(CudaArray<Material>::size_type(geometry.materialBuffer.size()));
	
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

void CudaRenderer::_buildKdTree(const Obj::GeometryContainer &geometry, CudaArray<Types::Node> &nodeBuffer, CudaArray<Types::Triangle> &triangleBuffer, CudaArray<Types::Mesh> &meshBuffer, CudaArray<Material> &materialBuffer)
{
	std::vector<Cuda::Types::Node>		nodes;
	std::vector<Cuda::Types::Triangle>	triangles;
	std::vector<Cuda::Types::Mesh>		meshes;
	
	for (uint32_t meshIndex = 0; meshIndex < geometry.meshBuffer.size(); meshIndex++)
	{
		const Obj::Mesh		&objMesh	= geometry.meshBuffer[meshIndex];
		Cuda::Types::Mesh	mesh;
		mesh.materialOffset				= objMesh.materialOffset;
		meshes.push_back(mesh);
	}
	
	KdTreeBuilder builder;
	builder.build(geometry);
	
	// Traverse tree
	this->_traverseKdTree(geometry, builder.root(), nodes, triangles);
	
	nodeBuffer		= CudaArray<Cuda::Types::Node>(CudaArray<Cuda::Types::Node>::size_type(nodes.size()));
	triangleBuffer	= CudaArray<Cuda::Types::Triangle>(CudaArray<Cuda::Types::Triangle>::size_type(triangles.size()));
	meshBuffer		= CudaArray<Cuda::Types::Mesh>(CudaArray<Cuda::Types::Mesh>::size_type(meshes.size()));
	materialBuffer	= CudaArray<Material>(CudaArray<Material>::size_type(geometry.materialBuffer.size()));
	
	for (CudaArray<Cuda::Types::Node>::size_type i = 0; i < nodes.size(); i++)
	{
		nodeBuffer[i] = nodes[i];
	}
	
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

void CudaRenderer::_traverseKdTree(
	const Obj::GeometryContainer	&geometry,
	const Node						*node,
	std::vector<Types::Node>		&deviceNodes,
	std::vector<Types::Triangle>	&deviceTriangles)
{
	std::stack<const Node *, std::vector<const Node *>> stack;
	uint32_t ownIndex = 0;
	
	deviceNodes.resize(KdTreeBuilder::treeSize(node));
	
	stack.push(node);
	
	while (!stack.empty())
	{
		const Node *parent = stack.top();
		
		stack.pop();
		
		deviceNodes[ownIndex].axis			= parent->axis;
		deviceNodes[ownIndex].boundingBox	= parent->boundingBox;
		
		if (!parent->leafs.empty())
		{
			assert(parent->left.get()	!= nullptr);
			assert(parent->right.get()	!= nullptr);
			
			uint32_t rightNodeIndex	= ownIndex + KdTreeBuilder::treeSize(parent->left.get()) + 1;
			uint32_t leftNodeIndex	= ownIndex + 1;
			
			deviceNodes[ownIndex].rightNodeIndex		= rightNodeIndex;
			deviceNodes[rightNodeIndex].parentNodeIndex	= ownIndex;
			stack.push(parent->right.get());
			
			deviceNodes[ownIndex].leftNodeIndex			= leftNodeIndex;
			deviceNodes[leftNodeIndex].parentNodeIndex	= ownIndex;
			stack.push(parent->left.get());
		}
		else
		{
			deviceNodes[ownIndex].leafBeginIndex = uint32_t(deviceTriangles.size());
			
			for (const Leaf &leaf : parent->leafs)
			{
				const Obj::Mesh		&objMesh	= geometry.meshBuffer[leaf.meshIndex];
				const Obj::Triangle &triangle	= geometry.triangleBuffer[objMesh.triangleOffset + leaf.triangleIndex];
				
				Math::Vector4 v0, v1, v2, e01, e02, e12, n0, n1, n2;
				
				v0	= geometry.vertexBuffer[triangle.vertices[0]];
				v1	= geometry.vertexBuffer[triangle.vertices[1]];
				v2	= geometry.vertexBuffer[triangle.vertices[2]];
				e01	= v1 - v0;
				e02	= v2 - v0;
				e12	= v2 - v1;
				
				n0	= geometry.normalBuffer[triangle.normals[0]];
				n1	= geometry.normalBuffer[triangle.normals[1]];
				n2	= geometry.normalBuffer[triangle.normals[2]];
				
				deviceTriangles.push_back(Cuda::Types::Triangle{v0, e01, e02, e12, n0, n1, n2, leaf.meshIndex});
			}
			
			deviceNodes[ownIndex].leafEndIndex = uint32_t(deviceTriangles.size() - 1);
		}
		
		ownIndex++;
	}
}

}
