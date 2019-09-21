#include "debugstream.h"

namespace Utility
{

const std::string DebugStream::standard			= "\x1B[0m";
const std::string DebugStream::red				= "\x1B[31m";
const std::string DebugStream::green			= "\x1B[32m";
const std::string DebugStream::yellow			= "\x1B[33m";
const std::string DebugStream::blue				= "\x1B[34m";
const std::string DebugStream::magenta			= "\x1B[35m";
const std::string DebugStream::cyan				= "\x1B[36m";

DebugStream::DebugStream()
{
}

DebugStream::~DebugStream()
{
	*this << standard << "\n";
	std::cout << this->_stream.str();
}

}
