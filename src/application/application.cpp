#include <QDebug>

#include <color.h>

#include "application.h"

namespace PathTracer
{

Application::Application(QWidget *parent) : QMainWindow(parent)
{
	this->_scrollArea = new QScrollArea();
	this->_imageLabel = new QLabel();
	this->_toolBar = new QToolBar();
	this->_progressBar = new QProgressBar();
	
	this->_scrollArea->setWidget(this->_imageLabel);
	this->_scrollArea->setWidgetResizable(true);
	this->_imageLabel->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
//	this->_imageLabel->setScaledContents(true);
	this->setCentralWidget(this->_scrollArea);
	
	this->_toolBar->addWidget(this->_progressBar);
	this->addToolBar(Qt::ToolBarArea::BottomToolBarArea, this->_toolBar);
	
	this->resize(640, 480);
	
	this->_initializeScene();
}

Application::~Application()
{
//	this->_renderThread->quit();
//	this->_renderThread->wait();
	this->_renderThread->terminate();
}

void Application::render(const uint32_t width, const uint32_t height, const float fieldOfView, const uint32_t samples, const uint32_t bounces)
{
//	this->resize(int(width), int(height));
	this->_progressBar->setRange(0, int(samples));
	this->_progressBar->setValue(0);
	
	this->_image = QImage(int(width), int(height), QImage::Format::Format_RGB888);
	this->_image.fill(Qt::GlobalColor::black);
	this->_frameBuffer = {width, height};
	this->_frameBuffer.registerCallBack([this](const uint32_t x, const uint32_t y){
		emit this->dataAvailable(x, y);
	});
	
	this->_renderThread = new RenderThread(&this->_frameBuffer, &this->_geometry, fieldOfView, samples, bounces, this);
	
//	connect(this, &Application::dataAvailable, this, &Application::_updatePixel);
	connect(this->_renderThread, &RenderThread::dataAvailable, this, &Application::_updateImage);
	
	this->_renderThread->start();
}

void Application::_updatePixel(const quint32 x, const quint32 y)
{
	Rendering::Color color = Rendering::Color::fromVector4(this->_frameBuffer.pixel(x, y));
	this->_image.setPixel(int(x), int(y), qRgb(color.red(), color.green(), color.blue()));
	
	this->_imageLabel->setPixmap(QPixmap::fromImage(this->_image));
}

void Application::_updateImage()
{
	for (uint32_t h = 0; h < this->_frameBuffer.height(); h++)
	{
		for (uint32_t w = 0; w < this->_frameBuffer.width(); w++)
		{
			Rendering::Color color = Rendering::Color::fromVector4(this->_frameBuffer.pixel(w, h));
			this->_image.setPixel(int(w), int(h), qRgb(color.red(), color.green(), color.blue()));
		}
	}
	
	this->_imageLabel->setPixmap(QPixmap::fromImage(this->_image));
	
	this->_progressBar->setValue(this->_progressBar->value() + 1);
}

void Application::_initializeScene()
{
	Rendering::Material red{{1.0f, 0.0f, 0.0f}};
	Rendering::Material green{{0.0f, 1.0f, 0.0f}};
	Rendering::Material blue{{0.0f, 0.0f, 0.7f}};
	Rendering::Material cyan{{0.0f, 0.7f, 0.7f}};
	Rendering::Material magenta{{0.7f, 0.0f, 0.7f}};
	Rendering::Material yellow{{1.0f, 1.0f, 0.0f}};
	Rendering::Material black{{0.0f, 0.0f, 0.0f}};
	Rendering::Material halfWhite{{1.0f, 1.0f, 1.0f}};
	Rendering::Material white{{1.0f, 1.0f, 1.0f}};
	Rendering::Material halfGrey{{0.9f, 0.9f, 0.9f}};
	Rendering::Material grey{{0.2f, 0.2f, 0.2f}};
	Rendering::Material whiteLight{{1.0f, 1.0f, 1.0f}, 8.0f};
	Rendering::Material cyanLight{{0.0f, 1.0f, 1.0f}, 1.0f};
	
	this->_geometry.materialBuffer = {red, green, blue, cyan, magenta, yellow, black, white, halfGrey, grey, whiteLight, cyanLight};
	
	Rendering::Obj::Mesh cube0 = Rendering::Obj::Mesh::cube(1, 3, this->_geometry);
	cube0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f), this->_geometry);
	cube0.translate({-1.5f, -0.5f, -4.5f}, this->_geometry);
	
	Rendering::Obj::Mesh cube1 = Rendering::Obj::Mesh::cube(1, 4, this->_geometry);
	cube1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / -4.0f), this->_geometry);
	cube1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / -4.0f), this->_geometry);
	cube1.translate({2.5f, 0.2f, -5.5f}, this->_geometry);
	
//	Rendering::Obj::Mesh sphere = Rendering::Obj::Mesh::sphere(1.0f, 16, 8, 2, this->_geometry);
//	sphere.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 4.0f), this->_geometry);
//	sphere.translate({0.0f, 0.0f, -5.0f}, this->_geometry);
	
	Rendering::Obj::Mesh lightPlane0 = Rendering::Obj::Mesh::plane(4.0f, 10, this->_geometry);
	lightPlane0.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 2.0f), this->_geometry);
	lightPlane0.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 4.0f), this->_geometry);
	lightPlane0.translate({-5.5f, 1.5f, -9.0f}, this->_geometry);
	
//	Rendering::Obj::Mesh lightPlane1 = Rendering::Obj::Mesh::plane(2.0f, 11, this->_geometry);
//	lightPlane1.transform(Math::Matrix4x4::rotationMatrixX(float(M_PI) / 2.0f), this->_geometry);
//	lightPlane1.transform(Math::Matrix4x4::rotationMatrixY(float(M_PI) / 1.0f), this->_geometry);
//	lightPlane1.translate({0.0f, 0.0f, -1.0f}, this->_geometry);
	
	Rendering::Obj::Mesh worldCube = Rendering::Obj::Mesh::cube(20, 8, this->_geometry);
	worldCube.invert(this->_geometry);
	worldCube.translate({-2.0f, 9.0f, -2.0f}, this->_geometry);
	
	this->_geometry.meshBuffer.push_back(cube0);
	this->_geometry.meshBuffer.push_back(cube1);
//	this->_geometry.meshBuffer.push_back(sphere);
	this->_geometry.meshBuffer.push_back(lightPlane0);
//	this->_geometry.meshBuffer.push_back(lightPlane1);
	this->_geometry.meshBuffer.push_back(worldCube);
}

}
