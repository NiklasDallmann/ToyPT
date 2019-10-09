# ToyPT

ToyPT is a research project. The goal is to write a pathtracer in C++ with kernels using AVX, CUDA and OpenCL.

## Building

Building this project is straight forward.

```bash
mkdir <build directory>
cd <build directory>
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<installation directory> <path to repository>
make -j<threads> -l<cores>
[sudo] make install
```

## Dependencies

ToyPT depends on OpenMP, OpenCL, Qt, Open Image Deniose, CUDA and CXXUtility.
