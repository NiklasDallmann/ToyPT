#include "renderthread.h"

namespace PathTracer
{

RenderThread::RenderThread(Rendering::FrameBuffer *frameBuffer, Rendering::Obj::GeometryContainer *geometry, const float fieldOfView, const uint32_t samples, const uint32_t bounces, QObject *parent) :
	QThread(parent),
	_abort(false),
	_fieldOfView(fieldOfView),
	_samples(samples),
	_bounces(bounces),
	_frameBuffer(frameBuffer),
	_geometry(geometry)
{
}

RenderThread::~RenderThread()
{
}

void RenderThread::run()
{
	this->_abort = false;
	
	this->_renderer.render(*this->_frameBuffer, *this->_geometry,
		[this](){
			emit this->dataAvailable();
		},
	this->_abort, this->_fieldOfView, this->_samples, this->_bounces);
}

void RenderThread::quit()
{
	this->_abort = true;
}

}
