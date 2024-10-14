cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(cuda_example CUDA CXX) 

# Find CUDA
find_package(CUDA REQUIRED)

# Add an executable
add_executable(sort_array sort_array.cu)

# Specify CUDA target properties
set_target_properties(sort_array PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)