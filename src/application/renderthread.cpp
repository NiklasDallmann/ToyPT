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
	
	switch (this->_imageType)
	{
		case ImageType::Color:
			this->_renderer.render(*this->_frameBuffer, *this->_geometry,
				[this](){
					emit this->dataAvailable();
				},
			this->_abort, this->_fieldOfView, this->_samples, this->_bounces, {1.0f, 1.0f, 1.0f});
			break;
		case ImageType::Albedo:
			this->_renderer.renderAlbedoMap(*this->_frameBuffer, *this->_geometry, this->_fieldOfView);
			break;
		case ImageType::Normal:
			this->_renderer.renderNormalMap(*this->_frameBuffer, *this->_geometry, this->_fieldOfView);
			break;
	}
}

void RenderThread::configure(Rendering::FrameBuffer *frameBuffer, Rendering::Obj::GeometryContainer *geometry, const float fieldOfView, const uint32_t samples, const uint32_t bounces, const ImageType imageType)
{
	this->_frameBuffer = frameBuffer;
	this->_geometry = geometry;
	this->_fieldOfView = fieldOfView;
	this->_samples = samples;
	this->_bounces = bounces;
	this->_imageType = imageType;
}

void RenderThread::quit()
{
	this->_abort = true;
}

void RenderThread::_onFinished()
{
	switch (this->_imageType)
	{
		case ImageType::Color:
			emit colorMapFinished();
			break;
		case ImageType::Albedo:
			emit albedoMapFinished();
			break;
		case ImageType::Normal:
			emit normalMapFinished();
			break;
	}
}

}
