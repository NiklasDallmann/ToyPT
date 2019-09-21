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
	
	this->_renderThread.configure(&this->_frameBuffer, &this->_geometry, fieldOfView, samples, bounces, tileSize);
	
	this->_renderThread.start();
}

void Application::_updatePixel(const quint32 x, const quint32 y)
{
	Rendering::Color color = Rendering::Color::fromVector4(this->_frameBuffer.pixel(x, y));
	this->_image.setPixel(int(x), int(y), qRgb(color.red(), color.green(), color.blue()));
	
	QPixmap pixmap = QPixmap::fromImage(this->_image);
	int width = this->_imageLabel->width();
	int height = this->_imageLabel->height();
	
	this->_imageLabel->setPixmap(pixmap.scaled(width, height, Qt::KeepAspectRatio));
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
	
	this->addToolBar(Qt::ToolBarArea::RightToolBarArea, this->_toolBar);
	
	this->resize(1024, 480);
}

void Application::_doConnects()
{
	connect(this->_widthInput, &QLineEdit::textEdited, [this]()
	{
		bool success;
		uint32_t newValue = this->_widthInput->text().toUInt(&success);
		
		if (success)
		{
			this->_settings.width = newValue;
		}
		else
		{
			this->_widthInput->setText(QString::number(this->_settings.width));
		}
	});
	
	connect(this->_heightInput, &QLineEdit::textEdited, [this]()
	{
		bool success;
		uint32_t newValue = this->_heightInput->text().toUInt(&success);
		
		if (success)
		{
			this->_settings.height = newValue;
		}
		else
		{
			this->_heightInput->setText(QString::number(this->_settings.height));
		}
	});
	
	connect(this->_fovInput, &QLineEdit::textEdited, [this]()
	{
		bool success;
		float newValue = this->_fovInput->text().toFloat(&success);
		
		if (success)
		{
			this->_settings.fieldOfView = newValue;
		}
		else
		{
			this->_fovInput->setText(QString::number(double(this->_settings.fieldOfView)));
		}
	});
	
	connect(this->_samplesInput, &QLineEdit::textEdited, [this]()
	{
		bool success;
		uint32_t newValue = this->_samplesInput->text().toUInt(&success);
		
		if (success)
		{
			this->_settings.samples = newValue;
		}
		else
		{
			this->_samplesInput->setText(QString::number(this->_settings.samples));
		}
	});
	
	connect(this->_bouncesInput, &QLineEdit::textEdited, [this]()
	{
		bool success;
		uint32_t newValue = this->_bouncesInput->text().toUInt(&success);
		
		if (success)
		{
			this->_settings.bounces = newValue;
		}
		else
		{
			this->_bouncesInput->setText(QString::number(this->_settings.bounces));
		}
	});
	
	connect(this->_tileSizeInput, &QLineEdit::textEdited, [this]()
	{
		bool success;
		uint32_t newValue = this->_tileSizeInput->text().toUInt(&success);
		
		if (success)
		{
			this->_settings.tileSize = newValue;
		}
		else
		{
			this->_tileSizeInput->setText(QString::number(this->_settings.tileSize));
		}
	});
	
	connect(this->_startRenderButton, &QPushButton::clicked, [this]()
	{
		if (this->_renderThread.isRunning())
		{
			this->_renderThread.quit();
			this->_renderThread.wait();
		}
		
		this->render(this->_settings.width, this->_settings.height, this->_settings.fieldOfView, this->_settings.samples, this->_settings.bounces, this->_settings.tileSize);
	});
	
	connect(this->_stopRenderButton, &QPushButton::clicked, [this]()
	{
		if (this->_renderThread.isRunning())
		{
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
}

void Application::_initializeScene()
{
	Rendering::Material red{{1.0f, 0.0f, 0.0f}};
	Rendering::Material green{{0.0f, 1.0f, 0.0f}};
	Rendering::Material blue{{0.0f, 0.0f, 1.0f}, 0.0f, 0.3f};
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
	
	Rendering::Obj::Mesh cube0 = Rendering::Obj::Mesh::cube(1, 3, this->_geometry);
	cube0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f), this->_geometry);
	cube0.translate({-1.5f, -0.5f, -4.0f}, this->_geometry);
	
	Rendering::Obj::Mesh cube1 = Rendering::Obj::Mesh::cube(1, 4, this->_geometry);
	cube1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / -4.0f), this->_geometry);
	cube1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / -4.0f), this->_geometry);
	cube1.translate({2.5f, 0.2f, -5.5f}, this->_geometry);
	
//	Rendering::Obj::Mesh sphere = Rendering::Obj::Mesh::sphere(1.0f, 16, 8, 2, this->_geometry);
//	sphere.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 4.0f), this->_geometry);
//	sphere.translate({0.0f, 0.0f, -5.0f}, this->_geometry);
	
	Rendering::Obj::Mesh lightPlane0 = Rendering::Obj::Mesh::plane(8.0f, 10, this->_geometry);
	lightPlane0.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 1.0f), this->_geometry);
	lightPlane0.translate({-0.5f, 3.0f, -4.5f}, this->_geometry);
	
//	Rendering::Obj::Mesh plane = Rendering::Obj::Mesh::plane(50.0f, 9, this->_geometry);
//	plane.translate({0.0f, -1.0f, 0.0f}, this->_geometry);
	
	Rendering::Obj::Mesh worldCube = Rendering::Obj::Mesh::cube(20, 9, this->_geometry);
	worldCube.invert(this->_geometry);
	worldCube.translate({-2.0f, 9.0f, -2.0f}, this->_geometry);
	
	this->_geometry.meshBuffer.push_back(cube0);
	this->_geometry.meshBuffer.push_back(cube1);
//	this->_geometry.meshBuffer.push_back(sphere);
	this->_geometry.meshBuffer.push_back(lightPlane0);
//	this->_geometry.meshBuffer.push_back(plane);
	this->_geometry.meshBuffer.push_back(worldCube);
}

}
