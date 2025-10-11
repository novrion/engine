cmake_minimum_required(VERSION 3.29)
#set(CMAKE_CXX_SCAN_FOR_MODULES ON)
project(engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CSS_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package (glfw3 REQUIRED)

add_executable(engine src/engine.cpp)
target_link_libraries(engine PRIVATE
    Vulkan::Vulkan
    glfw
)

