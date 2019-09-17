#include "renderthread.h"

namespace PathTracer
{

RenderThread::RenderThread(QObject *parent) :
	QThread(parent)
{
	connect(this, &RenderThread::finished, this, &RenderThread::_onFinished);
}

RenderThread::~RenderThread()
{
}

void RenderThread::run()
{
	this->_abort = false;
	
	this->_renderer.render(*this->_frameBuffer, *this->_geometry,
	   [this](){
		   emit this->tileFinished();
	   },
	this->_abort, this->_fieldOfView, this->_samples, this->_bounces, this->_tileSize, {1.0f, 1.0f, 1.0f});
}

void RenderThread::configure(Rendering::FrameBuffer *frameBuffer, Rendering::Obj::GeometryContainer *geometry, const float fieldOfView, const uint32_t samples,
							 const uint32_t bounces, const uint32_t tileSize)
{
	this->_frameBuffer = frameBuffer;
	this->_geometry = geometry;
	this->_fieldOfView = fieldOfView;
	this->_samples = samples;
	this->_bounces = bounces;
	this->_tileSize = tileSize;
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
