cmake_minimum_required(VERSION 3.25)
project(tube_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_OSX_ARCHITECTURES arm64)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(tube_detector
    src/main.cpp
    src/Detector.cpp
)

target_link_libraries(tube_detector
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

find_package(OpenCV REQUIRED PATHS /opt/homebrew/opt/opencv)
