#include <stddef.h>
#include <simd/simdrenderer.h>
#include <cuda/cudarenderer.h>

#include "renderthread.h"

namespace ToyPT
{

RenderThread::RenderThread(QObject *parent) :
	QThread(parent)
{
//	this->_renderer = std::make_unique<Rendering::SimdRenderer>();
	this->_renderer = std::make_unique<Rendering::Cuda::CudaRenderer>();
	connect(this, &RenderThread::finished, this, &RenderThread::_onFinished);
}

RenderThread::~RenderThread()
{
}

void RenderThread::run()
{
	this->_abort = false;
	
	this->_renderer->render(
		*this->_frameBuffer,
		this->_settings,
		*this->_geometry,
		[this](const uint32_t x0, const uint32_t y0, const uint32_t x1, const uint32_t y1){
			emit this->tileFinished(x0, y0, x1, y1);
		},
		this->_abort
	);
	
	emit this->tileFinished(0, 0, this->_frameBuffer->width(), this->_frameBuffer->height());
}

void RenderThread::configure(Rendering::FrameBuffer *frameBuffer, const Rendering::RenderSettings &settings, const Rendering::Obj::GeometryContainer *geometry)
{
	this->_frameBuffer	= frameBuffer;
	this->_settings		= settings;
	this->_geometry		= geometry;
}

void RenderThread::quit()
{
	this->_abort = true;
}

void RenderThread::_onFinished()
{
	emit renderingFinished();
}

}
