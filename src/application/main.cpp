#include <iostream>

#include <algorithm>
#include <framebuffer.h>
#include <math.h>
#include <math/matrix4x4.h>
#include <mesh.h>
#include <triangle.h>
#include <vector>
#include <math/vector4.h>

#include <QApplication>

#include "application.h"

int main(int argc, char **argv)
{
	QApplication qtApp(argc, argv);
	QApplication::setQuitOnLastWindowClosed(true);
	ToyPT::Application app;
	
	app.show();
	
	qtApp.exec();
	
	return 0;
}
