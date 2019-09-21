#include <stdint.h>
#include <cmath>
#include <fstream>
#include <random>

#include "utility/debugstream.h"
#include "openclrenderer.h"
#include "randomnumbergenerator.h"

namespace Rendering
{

OpenCLRenderer::OpenCLRenderer() :
	AbstractRenderer()
{
	this->_initializeHardware();
	this->_buildKernel();
}

void OpenCLRenderer::render(FrameBuffer &frameBuffer, Obj::GeometryContainer &geometry, const OpenCLRenderer::CallBack &callBack, const bool &abort,
					  const float fieldOfView, const uint32_t samples, const uint32_t bounces, const uint32_t tileSize, const Math::Vector4 &skyColor)
{
	const uint32_t width = frameBuffer.width();
	const uint32_t height = frameBuffer.height();
	float fovRadians = fieldOfView / 180.0f * float(M_PI);
	float zCoordinate = -(width/(2.0f * std::tan(fovRadians / 2.0f)));
	
	Storage::geometryToBuffer(geometry, this->_triangleBuffer, this->_meshBuffer);
	
	std::random_device device;
	const uint32_t tilesVertical = height / tileSize + ((height % tileSize) > 0);
	const uint32_t tilesHorizontal = width / tileSize + ((width % tileSize) > 0);
	
	for (uint32_t tileVertical = 0; tileVertical < tilesVertical; tileVertical++)
	{
		for (uint32_t tileHorizontal = 0; tileHorizontal < tilesHorizontal; tileHorizontal++)
		{
			uint32_t startVertical = tileSize * tileVertical;
			uint32_t startHorizontal = tileSize * tileHorizontal;
			uint32_t endVertical = std::min(startVertical + tileSize, height);
			uint32_t endHorizontal = std::min(startHorizontal + tileSize, width);
			
			for (uint32_t h = startVertical; h < endVertical; h++)
			{
				for (uint32_t w = startHorizontal; (w < endHorizontal) & !abort; w++)
				{
					RandomNumberGenerator rng(device());
					Math::Vector4 color;
					
					for (size_t sample = 1; sample <= samples; sample++)
					{
						float offsetX, offsetY;
						const float scalingFactor = 1.0f / float(std::numeric_limits<uint32_t>::max());
						offsetX = rng.get(scalingFactor)  - 0.5f;
						offsetY = rng.get(scalingFactor) - 0.5f;
						
						float x = (w + offsetX + 0.5f) - (width / 2.0f);
						float y = -(h + offsetY + 0.5f) + (height / 2.0f);
						
						Math::Vector4 direction{x, y, zCoordinate};
						direction.normalize();
					
//						color += this->_castRay({{0, 0, 0}, direction}, geometry, rng, bounces, skyColor);
					}
					
					frameBuffer.setPixel(w, h, (color / float(samples)));
				}
			}
			
//			if (!abort)
//			{
//				callBack();
//			}
		}
	}
}

void OpenCLRenderer::_initializeHardware()
{
	cl_int error = CL_SUCCESS;
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	
	// Select OpenCL platform
	cl::Platform::get(&platforms);
	
	infoLog << "Available OpenCL platforms:";
	
	for (cl::Platform platform : platforms)
	{
		debugLog << "\t" << platform.getInfo<CL_PLATFORM_NAME>();
	}
	
	this->_platform = platforms.front();
	
	infoLog << "Selected platform: " << this->_platform.getInfo<CL_PLATFORM_NAME>();
	
	// Select OpenCL device
	this->_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	
	infoLog << "Available OpenCL devices:";
	
	for (cl::Device device : devices)
	{
		debugLog << "\t" << device.getInfo<CL_DEVICE_NAME>();
	}
	
	this->_device = devices.front();
	
	infoLog << "Selected device: " << this->_device.getInfo<CL_DEVICE_NAME>();
	
	this->_context = cl::Context(this->_device, nullptr, nullptr, nullptr, &error);
}

bool OpenCLRenderer::_buildKernel()
{
	bool returnValue = true;
	
	std::fstream stream(KERNEL_LOCATION);
	std::string source;
	cl_int error = CL_SUCCESS;
	
	this->_program = cl::Program(this->_context, source);
	error = this->_program.build({this->_device});
	
	const std::string buildLog = this->_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->_device);
	
	if (error == CL_SUCCESS)
	{
		infoLog << "Build successfull";
	}
	else
	{
		errorLog << "Build failed";
		debugLog << buildLog;
		returnValue = false;
	}
	
	return returnValue;
}

}
