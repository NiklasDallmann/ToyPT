#ifndef DEBUGSTREAM_H
#define DEBUGSTREAM_H

#include <iostream>
#include <sstream>
#include <string>

namespace Utility
{

class DebugStream
{
public:
	static const std::string standard;
	static const std::string red;
	static const std::string blue;
	static const std::string yellow;
	static const std::string green;
	static const std::string magenta;
	static const std::string cyan;
	
	DebugStream();
	~DebugStream();
	
	template <typename T>
	DebugStream &insert(const T &t)
	{
		this->_stream << t;
		return *this;
	}
	
private:
	std::stringstream _stream;
};

template <typename T>
DebugStream &operator<<(DebugStream &&debug, const T &t)
{
	return debug.insert(t);
}

template <typename T>
DebugStream &operator<<(DebugStream &debug, const T &t)
{
	return debug.insert(t);
}

} // namespace Utility

#define debugLog		Utility::DebugStream() << "DEBUG	"
#define warningLog		Utility::DebugStream() << Utility::DebugStream::yellow << "WARN	" << Utility::DebugStream::standard
#define errorLog		Utility::DebugStream() << Utility::DebugStream::red << "ERROR	" << Utility::DebugStream::standard
#define traceLog		Utility::DebugStream() << Utility::DebugStream::magenta << "TRACE	" << Utility::DebugStream::standard
#define infoLog			Utility::DebugStream() << Utility::DebugStream::cyan << "INFO	" << Utility::DebugStream::standard

#endif // DEBUGSTREAM_H
