#include <QDebug>
#include <QString>

#include <color.h>
#include <debugstream.h>

#include "application.h"

namespace PathTracer
{

Application::Application(QWidget *parent) : QMainWindow(parent)
{
	this->_buildUi();
	
	this->_doConnects();
	
	this->_initializeScene();
}

Application::~Application()
{
	this->_renderThread.quit();
	this->_renderThread.wait();
}

void Application::render(const uint32_t width, const uint32_t height, const float fieldOfView, const uint32_t samples, const uint32_t bounces, const uint32_t tileSize)
{
	const uint32_t tilesVertical = height / tileSize + ((height % tileSize) > 0);
	const uint32_t tilesHorizontal = width / tileSize + ((width % tileSize) > 0);
	
	this->_progressBar->setRange(0, int(tilesVertical * tilesHorizontal));
	this->_progressBar->setValue(0);
//	this->_progressBar->setFormat(QStringLiteral("%v/%m samples"));
	
	this->_image = QImage(int(width), int(height), QImage::Format::Format_RGB888);
	this->_image.fill(Qt::GlobalColor::black);
	this->_frameBuffer = {width, height};
	this->_onTileFinished(0, 0, this->_frameBuffer.width(), this->_frameBuffer.height());
	
	connect(&this->_timeUpdateTimer, &QTimer::timeout, this, &Application::_onTimeUpdate);
	this->_renderTime = {};
	this->_renderTime.start();
	this->_timeUpdateTimer.start(30);
	
	this->_renderThread.configure(&this->_frameBuffer, &this->_geometry, &this->_lights, fieldOfView, samples, bounces, tileSize);
	this->_renderThread.start();
}

void Application::_onTileFinished(const uint32_t x0, const uint32_t y0, const uint32_t x1, const uint32_t y1)
{
	for (uint32_t h = y0; h < y1; h++)
	{
		for (uint32_t w = x0; w < x1; w++)
		{
			Rendering::Color color = Rendering::Color::fromVector4(this->_frameBuffer.pixel(w, h));
			this->_image.setPixel(int(w), int(h), qRgb(color.red(), color.green(), color.blue()));
		}
	}
	
	this->_updateImageLabel();
	
	this->_progressBar->setValue(this->_progressBar->value() + 1);
}

void Application::_onRenderFinished()
{
	disconnect(&this->_timeUpdateTimer, &QTimer::timeout, this, &Application::_onTimeUpdate);
}

void Application::_onDenoise()
{
	if (this->_renderThread.isRunning())
	{
		this->_renderThread.quit();
		this->_renderThread.wait();
	}
	
	this->_frameBuffer = Rendering::FrameBuffer::denoise(this->_frameBuffer);
	this->_onTileFinished(0, 0, this->_frameBuffer.width(), this->_frameBuffer.height());
}

void Application::_onTimeUpdate()
{
	int elapsed = this->_renderTime.elapsed();
	int msecs = elapsed % 1000;
	int secs = (elapsed / 1000) % 60;
	int mins = (elapsed / 60000) % 60;
	int hours = (elapsed / 3600000) % 60;
	QTime time(hours, mins, secs, msecs);
	this->_statusLabel->setText(time.toString(QStringLiteral("HH:mm:ss:zzz")));
}

void Application::resizeEvent(QResizeEvent *event)
{
	Q_UNUSED(event)
	this->_updateImageLabel();
}

void Application::_updateImageLabel()
{
	QPixmap pixmap = QPixmap::fromImage(this->_image);
	int width = this->_imageLabel->width();
	int height = this->_imageLabel->height();
	
	this->_imageLabel->setPixmap(pixmap.scaled(width, height, Qt::KeepAspectRatio));
}


void Application::_buildUi()
{
	this->_imageLabel = new QLabel();
	this->_scrollArea = new QScrollArea();
	
	this->_toolBar = new QToolBar();
	
	this->_renderSettingsLayout = new QVBoxLayout();
	this->_renderSettingsGroupbox = new QGroupBox();
	
	this->_widthInput = new QLineEdit();
	this->_heightInput = new QLineEdit();
	this->_fovInput = new QLineEdit();
	this->_samplesInput = new QLineEdit();
	this->_bouncesInput = new QLineEdit();
	this->_tileSizeInput = new QLineEdit();
	
	this->_startStopLayout = new QHBoxLayout();
	this->_startRenderButton = new QPushButton(QStringLiteral("Start"));
	this->_stopRenderButton = new QPushButton(QStringLiteral("Stop"));
	
	this->_saveButton = new QPushButton(QStringLiteral("Save"));
	this->_denoiseButton = new QPushButton(QStringLiteral("Denoise"));
	this->_progressBar = new QProgressBar();
	this->_statusLabel = new QLabel();
	
	this->_fileDialog = new QFileDialog(this);
	
	this->_fileDialog->setFileMode(QFileDialog::AnyFile);
	this->_fileDialog->setNameFilter(QStringLiteral("Supported formats (*.png *.PNG)"));
	this->_fileDialog->setAcceptMode(QFileDialog::AcceptSave);
	
	// Image view
	this->_scrollArea->setWidget(this->_imageLabel);
	this->_scrollArea->setWidgetResizable(true);
	this->_imageLabel->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
	this->setCentralWidget(this->_scrollArea);
	
	// Toolbar
	this->_startStopLayout->addWidget(this->_startRenderButton);
	this->_startStopLayout->addWidget(this->_stopRenderButton);
	
	this->_widthInput->setText(QString::number(this->_settings.width));
	this->_heightInput->setText(QString::number(this->_settings.height));
	this->_fovInput->setText(QString::number(double(this->_settings.fieldOfView)));
	this->_samplesInput->setText(QString::number(this->_settings.samples));
	this->_bouncesInput->setText(QString::number(this->_settings.bounces));
	this->_tileSizeInput->setText(QString::number(this->_settings.tileSize));
	
	this->_renderSettingsLayout->addWidget(this->_widthInput);
	this->_renderSettingsLayout->addWidget(this->_heightInput);
	this->_renderSettingsLayout->addWidget(this->_fovInput);
	this->_renderSettingsLayout->addWidget(this->_samplesInput);
	this->_renderSettingsLayout->addWidget(this->_bouncesInput);
	this->_renderSettingsLayout->addWidget(this->_tileSizeInput);
	this->_renderSettingsLayout->addLayout(this->_startStopLayout);
	
	this->_renderSettingsGroupbox->setTitle(QStringLiteral("Render Settings"));
	this->_renderSettingsGroupbox->setLayout(this->_renderSettingsLayout);
	
	this->_toolBar->addWidget(this->_renderSettingsGroupbox);
	this->_toolBar->addWidget(this->_saveButton);
	this->_toolBar->addWidget(this->_denoiseButton);
	this->_toolBar->addWidget(this->_progressBar);
	this->_toolBar->addWidget(this->_statusLabel);
	
	this->addToolBar(Qt::ToolBarArea::RightToolBarArea, this->_toolBar);
	
	this->resize(1024, 480);
}

void Application::_doConnects()
{
	connect(this->_startRenderButton, &QPushButton::clicked, [this]()
	{
		if (this->_renderThread.isRunning())
		{
			disconnect(&this->_timeUpdateTimer, &QTimer::timeout, this, &Application::_onTimeUpdate);
			this->_timeUpdateTimer.stop();
			this->_renderThread.quit();
			this->_renderThread.wait();
		}
		
		if (this->_applyRenderSettings())
		{
			this->render(this->_settings.width, this->_settings.height, this->_settings.fieldOfView, this->_settings.samples, this->_settings.bounces, this->_settings.tileSize);
		}
		else
		{
			this->_statusLabel->setText(QStringLiteral("Invalid parameters!"));
		}
	});
	
	connect(this->_stopRenderButton, &QPushButton::clicked, [this]()
	{
		if (this->_renderThread.isRunning())
		{
			disconnect(&this->_timeUpdateTimer, &QTimer::timeout, this, &Application::_onTimeUpdate);
			this->_timeUpdateTimer.stop();
			this->_renderThread.quit();
			this->_renderThread.wait();
		}
	});
	
	connect(this->_saveButton, &QPushButton::clicked, [this]()
	{
		this->_fileDialog->show();
	});
	
	connect(this->_denoiseButton, &QPushButton::clicked, this, &Application::_onDenoise);
	
	connect(this->_fileDialog, &QFileDialog::accepted, [this]()
	{
		QStringList selectedFiles = this->_fileDialog->selectedFiles();
		
		if (!selectedFiles.empty())
		{
			QString filename = selectedFiles[0];
			
			this->_image.save(filename, "png");
		}
	});
	
	connect(&this->_renderThread, &RenderThread::tileFinished, this, &Application::_onTileFinished);
	connect(&this->_renderThread, &RenderThread::finished, this, &Application::_onRenderFinished);
}

void Application::_initializeScene()
{
	Rendering::Material red{{1.0f, 0.0f, 0.0f}};
	Rendering::Material green{{0.0f, 1.0f, 0.0f}};
	Rendering::Material blue{{0.0f, 0.0f, 1.0f}, 0.0f, 0.5f};
	Rendering::Material cyan{{0.0f, 0.7f, 0.7f}};
	Rendering::Material magenta{{1.0f, 0.0f, 1.0f}, 0.0f, 1.0f};
	Rendering::Material yellow{{1.0f, 1.0f, 0.0f}};
	Rendering::Material black{{0.0f, 0.0f, 0.0f}};
	Rendering::Material halfWhite{{1.0f, 1.0f, 1.0f}};
	Rendering::Material white{{1.0f, 1.0f, 1.0f}};
	Rendering::Material halfGrey{{0.9f, 0.9f, 0.9f}};
	Rendering::Material grey{{0.8f, 0.8f, 0.8f}, 0.0f, 0.5f};
	Rendering::Material whiteLight{{1.0f, 1.0f, 1.0f}, 1.0f};
	Rendering::Material cyanLight{{0.0f, 1.0f, 1.0f}, 1.0f};
	Rendering::Material mirror{{0.0f, 0.0f, 0.0f}, 0.0f, 0.0f};
	
	//									0		1		2		3		4			5		6		7		8			9		10			11			12
	this->_geometry.materialBuffer = {	red,	green,	blue,	cyan,	magenta,	yellow,	black,	white,	halfGrey,	grey,	whiteLight,	cyanLight,	mirror};
	
	// Objects
	Rendering::Obj::Mesh cube0 = Rendering::Obj::Mesh::cube(1, 3, this->_geometry);
	cube0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f), this->_geometry);
	cube0.translate({-1.5f, -0.5f, -4.0f}, this->_geometry);
	
	Rendering::Obj::Mesh cube1 = Rendering::Obj::Mesh::cube(1, 4, this->_geometry);
	cube1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / -4.0f), this->_geometry);
	cube1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / -4.0f), this->_geometry);
	cube1.translate({2.5f, 0.2f, -5.5f}, this->_geometry);
	
	Rendering::Obj::Mesh sphere = Rendering::Obj::Mesh::sphere(1.0f, 16, 8, 12, this->_geometry);
	sphere.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 4.0f), this->_geometry);
	sphere.translate({0.0f, 0.0f, -5.0f}, this->_geometry);
	
	Rendering::Obj::Mesh worldCube = Rendering::Obj::Mesh::cube(20, 9, this->_geometry);
	worldCube.invert(this->_geometry);
	worldCube.translate({-2.0f, 9.0f, -2.0f}, this->_geometry);
	
	// Lights
	Rendering::Obj::Mesh lightPlane0 = Rendering::Obj::Mesh::plane(8.0f, 10, this->_geometry);
	lightPlane0.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 1.0f), this->_geometry);
	lightPlane0.translate({-0.5f, 3.0f, -4.5f}, this->_geometry);
	
	// Object buffer
	this->_geometry.meshBuffer.push_back(cube0);
	this->_geometry.meshBuffer.push_back(cube1);
	this->_geometry.meshBuffer.push_back(sphere);
	this->_geometry.meshBuffer.push_back(worldCube);
	
	// Light buffer
	this->_geometry.meshBuffer.push_back(lightPlane0);
}

bool Application::_applyRenderSettings()
{
	bool success;
	
	uint32_t integerValue;
	float floatValue;
	
	integerValue = this->_widthInput->text().toUInt(&success);
	if (!success)
	{
		goto error;
	}
	else
	{
		this->_settings.width = integerValue;
	}
	
	integerValue = this->_heightInput->text().toUInt(&success);
	if (!success)
	{
		goto error;
	}
	else
	{
		this->_settings.height = integerValue;
	}
	
	floatValue = this->_fovInput->text().toFloat(&success);
	if (!success)
	{
		goto error;
	}
	else
	{
		this->_settings.fieldOfView = floatValue;
	}
	
	integerValue = this->_samplesInput->text().toUInt(&success);
	if (!success)
	{
		goto error;
	}
	else
	{
		this->_settings.samples = integerValue;
	}
	
	integerValue = this->_bouncesInput->text().toUInt(&success);
	if (!success)
	{
		goto error;
	}
	else
	{
		this->_settings.bounces = integerValue;
	}
	
	integerValue = this->_tileSizeInput->text().toUInt(&success);
	if (!success)
	{
		goto error;
	}
	else
	{
		this->_settings.tileSize = integerValue;
	}
	
	return true;
	
	error:
	return false;
}

}
