#ifndef RENDERTHREAD_H
#define RENDERTHREAD_H

#include <stddef.h>

#include <QObject>
#include <QThread>

#include <framebuffer.h>
#include <geometrycontainer.h>
#include <simdrenderer.h>

namespace PathTracer
{

class RenderThread : public QThread
{
	Q_OBJECT
	
public:
	RenderThread(Rendering::FrameBuffer *frameBuffer, Rendering::Obj::GeometryContainer *geometry,
				 const float fieldOfView, const uint32_t samples, const uint32_t bounces,
				 QObject *parent = nullptr);
	virtual ~RenderThread() override;
	
	void run() override;
	
signals:
	void pixelAvailable(const uint32_t x, const uint32_t y);
	void dataAvailable();
	
public slots:
	void quit();
	
private:
	bool _abort;
	float _fieldOfView;
	uint32_t _samples;
	uint32_t _bounces;
	Rendering::FrameBuffer *_frameBuffer;
	Rendering::Obj::GeometryContainer *_geometry;
	Rendering::SimdRenderer _renderer;
};

} // namespace PathTracer

#endif // RENDERTHREAD_H
