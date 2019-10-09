#ifndef CUDAARRAY_H
#define CUDAARRAY_H

#include <vector>
#include <stdint.h>
#include <type_traits>

#include "cudaarrayprivate.h"

namespace ToyPT
{
namespace Rendering
{
namespace Cuda
{

///
/// Manages an array in the CUDA unified memory.
///
template <typename T>
class CudaArray
{
	static_assert(!std::is_reference<T>::value, "Reference types are not supported");
	static_assert(!std::is_pointer<T>::value, "Pointer types are not supported");
	static_assert(!std::is_const<T>::value, "Const qualified types are not supported");
	
public:
	using value_type				= T;
	using size_type					= uint32_t;
	using difference_type			= int32_t;
	using reference					= T&;
	using const_reference			= const T&;
	using pointer					= T*;
	using const_pointer				= const T*;
	using iterator					= pointer;
	using const_iterator			= const_pointer;
	
	CudaArray()
	{
	}
	
	///
	/// Constructs a CudaArray with memory for \a size elements.
	/// 
	/// The allocated memory is not initialized.
	///
	CudaArray(const size_type size) :
		_size(size)
	{
		allocateManagedCudaMemory(reinterpret_cast<void **>(&this->_data), size * sizeof (T));
	}
	
	///
	/// Destructs the object and frees the memory.
	///
	~CudaArray()
	{
		freeCudaMemory(this->_data);
	}
	
	size_type size() const
	{
		return this->_size;
	}
	
	pointer data()
	{
		return this->_data;
	}
	
	const_pointer data() const
	{
		return this->_data;
	}
	
	reference operator[](const size_type index)
	{
		return this->_data[index];
	}
	
	const_reference operator[](const size_type index) const
	{
		return this->_data[index];
	}
	
private:
	size_type _size = 0;
	pointer _data = nullptr;
};

} // namespace Cuda
} // namespace Rendering
} // namespace ToyPT

#endif // CUDAARRAY_H
