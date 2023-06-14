cmake_minimum_required(VERSION 3.0)
project(sof VERSION 1.1.0 LANGUAGES CXX)
# ビルドフラグの設定
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
# インクルードパスの追加
if(WIN32)
  include_directories(C:/local/include)
  link_directories(C:/local/lib)
else()
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
endif()
# ビルドに関する設定
set(EXECUTABLE "sof")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${EXECUTABLE} src/sof.cpp)
target_link_libraries(${EXECUTABLE} ${OpenCV_LIBS})
