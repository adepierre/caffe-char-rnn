cmake_minimum_required(VERSION 3.2)

project(caffe_char_rnn)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})

#Add Caffe
find_package(OpenCV)
find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

#Add source files
include_directories(include)

file(GLOB src_files "src/*.cpp")

file(GLOB header_files "include/*.h")

add_executable(caffe_char_rnn ${src_files} ${header_files})


target_link_libraries(caffe_char_rnn ${Caffe_LIBRARIES})