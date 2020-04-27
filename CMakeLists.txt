cmake_minimum_required(VERSION 3.16)
project(hl_rec)

set(CMAKE_CXX_STANDARD 17)

set(REPOS "/Users/eddiebarton/dev/repos")
set(LIBS "/Users/eddiebarton/dev/libs")
set(HALIDE_DISTRIB_DIR ${LIBS}/Halide)

include_directories(${HALIDE_DISTRIB_DIR}/include)
include_directories(${HALIDE_DISTRIB_DIR}/tools)
include_directories(${REPOS}/tinydngloader)
include_directories(include)

include(${HALIDE_DISTRIB_DIR}/halide.cmake)

add_definitions(-DTINY_DNG_LOADER_IMPLEMENTATION)
add_definitions(-DSTB_IMAGE_IMPLEMENTATION)
add_definitions(-DTINY_DNG_LOADER_ENABLE_ZIP)

add_executable(hl_rec src/hl_rec.cc
    src/image_io.cc
    src/color.cc
    ${REPOS}/tinydngloader/miniz.c
    ${REPOS}/tinydngloader/tiny_dng_loader.h
)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
  target_link_libraries(hl_rec PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(hl_rec PUBLIC ${HALIDE_DISTRIB_DIR}/bin/libHalide.dylib)
target_link_libraries(hl_rec PUBLIC /usr/local/Cellar/jpeg/9d/lib/libjpeg.dylib)
target_link_libraries(hl_rec PUBLIC /usr/local/Cellar/libpng/1.6.37/lib/libpng.dylib)
target_link_libraries(hl_rec PUBLIC /usr/local/Cellar/libtiff/4.1.0/lib/libtiff.dylib)

halide_generator(camera_pipe.generator SRCS src/camera_pipe_generator.cc)

foreach(AUTO_SCHEDULE false true)
  if(${AUTO_SCHEDULE})
    set(LIB camera_pipe_auto_schedule)
  else()
    set(LIB camera_pipe)
  endif()
  halide_library_from_generator(${LIB}
      GENERATOR camera_pipe.generator
      GENERATOR_ARGS auto_schedule=${AUTO_SCHEDULE})
  target_link_libraries(hl_rec PRIVATE ${LIB} ${curved_lib})
endforeach()