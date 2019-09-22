#ifndef RENDERTHREAD_H
#define RENDERTHREAD_H

#include <stddef.h>
#include <memory>

#include <QMetaType>
#include <QObject>
#include <QThread>

#include <abstractrenderer.h>
#include <framebuffer.h>
#include <geometrycontainer.h>
#include <openclrenderer.h>
#include <simdrenderer.h>

namespace PathTracer
{

class RenderThread : public QThread
{
	Q_OBJECT
	
public:
	RenderThread(QObject *parent = nullptr);
	virtual ~RenderThread() override;
	
	void run() override;
	
	void configure(Rendering::FrameBuffer *frameBuffer, Rendering::Obj::GeometryContainer *geometry, Rendering::Obj::GeometryContainer *lights,
				   const float fieldOfView, const uint32_t samples, const uint32_t bounces, const uint32_t tileSize);
	
signals:
	void tileFinished(const quint32 x0, const quint32 y0, const quint32 x1, const quint32 y1);
	void renderingFinished();
	
public slots:
	void quit();
	
private slots:
	void _onFinished();
	
private:
	bool _abort = false;
	float _fieldOfView = 0;
	uint32_t _samples = 0;
	uint32_t _bounces = 0;
	uint32_t _tileSize = 0;
	Rendering::FrameBuffer *_frameBuffer = nullptr;
	Rendering::Obj::GeometryContainer *_geometry = nullptr;
	Rendering::Obj::GeometryContainer *_lights = nullptr;
	std::unique_ptr<Rendering::AbstractRenderer> _renderer;
};

} // namespace PathTracer

#endif // RENDERTHREAD_H
