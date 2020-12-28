cmake_minimum_required(VERSION 3.15)

# unicode
add_definitions(-DUNICODE -D_UNICODE)
remove_definitions (-D_MBCS)

set(PROJECT_NAME FluidSolver)
project(${PROJECT_NAME})

add_subdirectory(CuSolver)
add_subdirectory(Renderer)