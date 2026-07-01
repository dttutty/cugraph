# libcugraph

`libcugraph` is the wheel wrapper for the native cuGraph C++/CUDA library under
`cpp`. It packages the low-level graph analytics runtime, including public
C/C++ headers, graph construction, algorithms, renumbering, shuffling,
sampling, and CMake package metadata used by downstream Python packages.

The wheel build compiles and installs the native library by default, even when
another `cugraph` CMake package is already visible in `CMAKE_PREFIX_PATH`. Set
`LIBCUGRAPH_PYTHON_USE_INSTALLED_CUGRAPH=ON` only for an explicit wrapper-only
build against an already installed native package.
