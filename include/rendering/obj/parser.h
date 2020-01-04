#ifndef OBJPARSER_H
#define OBJPARSER_H

#include <utility>
#include <stdint.h>
#include <string>
#include <vector>

#include <math/vector4.h>

#include "mesh.h"

namespace ToyPT::Rendering::Obj
{

class Parser
{
public:
	enum class VertexType
	{
		Geometric,
		Normal,
		Parameter,
		Texture
	};
	
	struct Vertex
	{
		VertexType				type			= VertexType::Geometric;
		Math::Vector4			coordinate		= {};
	};
	
	struct Face
	{
		std::vector<uint32_t>	vertexIndices	= {};
	};
	
private:
	std::vector<std::tuple<VertexType, Math::Vector4>>	_vertices;
	std::vector<Face>									_faces;
	std::vector<Mesh>									_meshes;
};

}

#endif // OBJPARSER_H
