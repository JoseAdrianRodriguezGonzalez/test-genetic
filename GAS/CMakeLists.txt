cmake_minimum_required(VERSION 3.16)
project(GA_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_BUILD_TYPE Debug)
# Incluir headers
include_directories(lib)

# Archivos fuente
set(SOURCES
    src/algorithms/elitist.cpp
    src/algorithms/gas.cpp
    src/benchmark/benchmark.cpp
)

# Crear librería
add_library(ga_lib ${SOURCES})

# Ejecutable
add_executable(main_experiment experiment/main1.cpp)

# Linkear
target_link_libraries(main_experiment ga_lib)
#target_compile_options(ga_lib PRIVATE -fsanitize=address -fno-omit-frame-pointer)
