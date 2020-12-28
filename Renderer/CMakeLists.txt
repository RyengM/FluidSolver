cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME Renderer)
project(${PROJECT_NAME} LANGUAGES CXX)

include_directories("../ThirdParty/OpenGL/include")
include_directories("../CuSolver/Include")

# the path is based on module vcxproj
link_libraries("../../ThirdParty/OpenGL/lib/glfw3")

file(GLOB_RECURSE HEADER_FILES "Source/*.h")
file(GLOB_RECURSE SOURCE_FILES "Source/*.cpp" "Source/*.c")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} CuSolver)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/Bin)

set(CMAKE_COMMAND xcopy /I /E /Y /D "\"${CMAKE_SOURCE_DIR}/Renderer/Shader\"" "\"${CMAKE_SOURCE_DIR}/Bin/$(Configuration)/Shader\"")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND})

set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/Bin/$(Configuration)")