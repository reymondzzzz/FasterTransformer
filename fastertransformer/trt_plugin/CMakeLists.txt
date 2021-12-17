set(SRC FasterTransformerPlugin.cpp FasterTransformerPlugin.h)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(TensorRT REQUIRED)

add_library(trt_plugin SHARED ${SRC})
target_include_directories(trt_plugin PRIVATE ${TENSORRT_INCLUDE_DIR})