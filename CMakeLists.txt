
set(CMAKE_CUDA_ARCHITECTURES 52)
project(sgemmCuda LANGUAGES  CUDA)

add_executable(sgemmCuda ./sgemm.cu)
