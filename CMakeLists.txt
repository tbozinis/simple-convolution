cmake_minimum_required(VERSION 3.12)
project(simple_convolution)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(simple_convolution
        cpu/Convolution.h
        cpu/main.cpp)
