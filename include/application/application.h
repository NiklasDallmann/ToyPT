#ifndef APPLICATION_H
#define APPLICATION_H

#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QMetaType>
#include <QObject>
#include <QProgressBar>
#include <QScrollArea>
#include <QToolBar>

#include <framebuffer.h>
#include <geometrycontainer.h>
#include <simdrenderer.h>

#include <renderthread.h>

namespace PathTracer
{

class Application : public QMainWindow
{
	Q_OBJECT
	
public:
	Application(QWidget *parent = nullptr);
	virtual ~Application();
	
	void render(const uint32_t width, const uint32_t height, const float fieldOfView, const uint32_t samples, const uint32_t bounces);
	
signals:
	void dataAvailable(const quint32 x, const quint32 y);
	
private slots:
	void _updatePixel(const quint32 x, const quint32 y);
	void _updateImage();
	
private:
	Rendering::Obj::GeometryContainer _geometry;
	Rendering::FrameBuffer _frameBuffer;
	
	QImage _image;
	QLabel *_imageLabel;
	QScrollArea *_scrollArea;
	QToolBar *_toolBar;
	QProgressBar *_progressBar;
	RenderThread *_renderThread;
	
	void _initializeScene();
};

} // namespace PathTracer

Q_DECLARE_METATYPE(uint32_t);

#endif // APPLICATION_H
