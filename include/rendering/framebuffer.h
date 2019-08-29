#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <functional>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "vector4.h"

namespace Rendering
{

class FrameBuffer
{
public:
	using CallBack = std::function<void(const uint32_t x, const uint32_t y)>;
	
	FrameBuffer(const uint32_t width = 0, const uint32_t height = 0);
	
	uint32_t width() const;
	uint32_t height() const;
	
	Math::Vector4 &pixel(const uint32_t x, const uint32_t y);
	const Math::Vector4 &pixel(const uint32_t x, const uint32_t y) const;
	void setPixel(const uint32_t x, const uint32_t y, const Math::Vector4 &color);
	bool save(const std::string &fileName);
	void registerCallBack(const CallBack callBack);
	void runCallBacks(const uint32_t x, const uint32_t y);
	
	static FrameBuffer denoise(const FrameBuffer &color, const FrameBuffer &normal);
	
private:
	uint32_t _width = 0;
	uint32_t _height = 0;
	std::vector<Math::Vector4> _buffer;
	std::vector<CallBack> _callbacks;
	
	static void _frameBufferToBuffer(const FrameBuffer &frameBuffer, std::vector<float> &buffer);
	static void _bufferToFrameBuffer(const std::vector<float> &buffer, FrameBuffer &frameBuffer);
};

} // namespace Rendering

#endif // FRAMEBUFFER_H
