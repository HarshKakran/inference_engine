cmake_minimum_required(VERSION 3.20)
project(inference_engine CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(llama REQUIRED)

add_executable(hello
    src/main.cpp
    src/sampling.cpp
    src/logging.cpp
)

target_link_libraries(hello PRIVATE llama)