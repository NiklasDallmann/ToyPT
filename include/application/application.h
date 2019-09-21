#ifndef APPLICATION_H
#define APPLICATION_H

#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMetaType>
#include <QObject>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QString>
#include <QToolBar>
#include <QVBoxLayout>

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
	virtual ~Application() override;
	
	void render(const uint32_t width, const uint32_t height, const float fieldOfView, const uint32_t samples, const uint32_t bounces, const uint32_t tileSize);
	
signals:
	void dataAvailable(const quint32 x, const quint32 y);
	
private slots:
	void _updatePixel(const quint32 x, const quint32 y);
	void _onTileFinished(const uint32_t x0, const uint32_t y0, const uint32_t x1, const uint32_t y1);
	void _onDenoise();
	
protected:
	virtual void resizeEvent(QResizeEvent *event) override;
	
private:
	struct RenderSettings
	{
		uint32_t width = 400;
		uint32_t height = 200;
		float fieldOfView = 70.0f;
		uint32_t samples = 32;
		uint32_t bounces = 3;
		uint32_t tileSize = 32;
	};
	
	Rendering::Obj::GeometryContainer _geometry;
	Rendering::FrameBuffer _frameBuffer;
	Rendering::FrameBuffer _albedoMap;
	Rendering::FrameBuffer _normalMap;
	RenderSettings _settings;
	
	QImage _image;
	QLabel *_imageLabel = nullptr;
	QScrollArea *_scrollArea = nullptr;
	
	QToolBar *_toolBar = nullptr;
	
	QVBoxLayout *_renderSettingsLayout = nullptr;
	QGroupBox *_renderSettingsGroupbox = nullptr;
	
	QLineEdit *_widthInput = nullptr;
	QLineEdit *_heightInput = nullptr;
	QLineEdit *_fovInput = nullptr;
	QLineEdit *_samplesInput = nullptr;
	QLineEdit *_bouncesInput = nullptr;
	QLineEdit *_tileSizeInput = nullptr;
	
	QHBoxLayout *_startStopLayout = nullptr;
	QPushButton *_startRenderButton = nullptr;
	QPushButton *_stopRenderButton = nullptr;
	QPushButton *_saveButton = nullptr;
	QPushButton *_denoiseButton = nullptr;
	
	QProgressBar *_progressBar = nullptr;
	
	QFileDialog *_fileDialog;
	
	RenderThread _renderThread;
	
	void _updateImageLabel();
	void _buildUi();
	void _doConnects();
	void _initializeScene();
};

} // namespace PathTracer

Q_DECLARE_METATYPE(uint32_t);

#endif // APPLICATION_H
