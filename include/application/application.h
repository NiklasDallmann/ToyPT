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
#include <QTime>
#include <QTimer>
#include <QToolBar>
#include <QVBoxLayout>

#include <rendering/framebuffer.h>
#include <rendering/geometrycontainer.h>
#include <rendering/rendersettings.h>

#include "renderthread.h"

namespace ToyPT
{

class Application : public QMainWindow
{
	Q_OBJECT
	
public:
	Application(QWidget *parent = nullptr);
	virtual ~Application() override;
	
	void render();
	
signals:
	void dataAvailable(const quint32 x, const quint32 y);
	
private slots:
	void _onTileFinished(const uint32_t x0, const uint32_t y0, const uint32_t x1, const uint32_t y1);
	void _onRenderFinished();
	void _onDenoise();
	void _onTimeUpdate();
	
protected:
	virtual void resizeEvent(QResizeEvent *event) override;
	
private:
	Rendering::Obj::GeometryContainer	_geometry;
	Rendering::FrameBuffer				_frameBuffer;
	Rendering::FrameBuffer				_albedoMap;
	Rendering::FrameBuffer				_normalMap;
	Rendering::RenderSettings			_settings;
	
	QTime								_renderTime;
	QTimer								_timeUpdateTimer;
	
	QImage								_image;
	QLabel								*_imageLabel				= nullptr;
	QScrollArea							*_scrollArea				= nullptr;
	
	QToolBar							*_toolBar					= nullptr;
	
	QVBoxLayout							*_renderSettingsLayout		= nullptr;
	QGroupBox							*_renderSettingsGroupbox	= nullptr;
	
	QLineEdit							*_widthInput				= nullptr;
	QLineEdit							*_heightInput				= nullptr;
	QLineEdit							*_fovInput					= nullptr;
	QLineEdit							*_samplesInput				= nullptr;
	QLineEdit							*_bouncesInput				= nullptr;
	QLineEdit							*_tileSizeInput				= nullptr;
	
	QHBoxLayout							*_startStopLayout			= nullptr;
	QPushButton							*_startRenderButton			= nullptr;
	QPushButton							*_stopRenderButton			= nullptr;
	QPushButton							*_saveButton				= nullptr;
	QPushButton							*_denoiseButton				= nullptr;
	
	QProgressBar						*_progressBar				= nullptr;
	QLabel								*_statusLabel				= nullptr;
	
	QFileDialog							*_fileDialog;
	
	RenderThread						_renderThread;
	
	void _updateImageLabel();
	void _buildUi();
	void _doConnects();
	void _initializeScene();
	bool _applyRenderSettings();
};

} // namespace ToyPT

Q_DECLARE_METATYPE(uint32_t);

#endif // APPLICATION_H
