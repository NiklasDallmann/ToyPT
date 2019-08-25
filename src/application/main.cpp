#include <iostream>

#include <algorithm>
#include <framebuffer.h>
#include <math.h>
#include <matrix4x4.h>
#include <mesh.h>
#include <simdrenderer.h>
#include <triangle.h>
#include <vector>
#include <vector4.h>

#include <QApplication>

#include "application.h"

int main(int argc, char **argv)
{
	std::cout << "Path Tracer" << std::endl;
	
	QApplication qtApp(argc, argv);
	QApplication::setQuitOnLastWindowClosed(true);
	PathTracer::Application app;
	
	app.show();
	//			width	height	FOV		samples	bounces
//	app.render(	400,	200,	70.0f,	1024,		3);
	
	qtApp.exec();
	
	return 0;
}
