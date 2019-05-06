cmake_minimum_required(VERSION 3.5.1)
project(6_1_RD_AlgoAgg)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES  Dewarper.cpp Dewarper_CPU.h main.cpp Scaler.cpp Assesors/assessorBase.h Assesors/rank_bluro.h Assesors/VOL.h
        Libs/RobustMatcher.h Libs/RobustMatcher.cpp Welder/Composer.h Libs/LinkedList.h Libs/ScoredImage.h Welder/Assessor.h)

find_package( OpenCV REQUIRED )

add_executable(6_1_RD_AlgoAgg ${SOURCE_FILES})


## Matlab cmake requries


set(Matlab_ROOT_DIR "/usr/local/MATLAB/MATLAB_Runtime/v92/")
set(MATLAB_FIND_DEBUG 1)

find_package(Matlab REQUIRED)
include_directories(${Matlab_INCLUDE_DIRS})

list(GET Matlab_LIBRARIES 0 Matlab_LIBRARY)
get_filename_component(Matlab_LIBRARY_DIR ${Matlab_LIBRARY} DIRECTORY)
SET(Matlab_LIBRARIES ${Matlab_LIBRARY_DIR}/../../runtime/glnxa64/libmwmclmcrrt.so)

link_directories(../Libs)
add_library(fish SHARED IMPORTED)
set_property(TARGET fish PROPERTY IMPORTED_LOCATION ../Libs/fish.so)

##

## Blend
#set(EIGEN_INCLUDE_DIR "../eigen" CACHE PATH "Where is the include directory of Eigen located")
#set(BLEND_TESTS_VERBOSE 0 CACHE BOOL "Tests will show visualizations when enabled")
#
#include_directories(${CMAKE_CURRENT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR} "inc")
#
## Library
#
#add_library(blend
#        inc/blend/poisson_solver.h
#        inc/blend/clone.h
#        inc/blend/blend.h
#        src/poisson_solver.cpp
#        src/clone.cpp
#        src/blend.cpp
#        )

#target_link_libraries(6_1_RD_AlgoAgg blend ${OpenCV_LIBS} )
##

#include_directories(Libs)

target_link_libraries( 6_1_RD_AlgoAgg ${OpenCV_LIBS} )

## Matlab cmake requries
target_link_libraries(6_1_RD_AlgoAgg fish ${Matlab_LIBRARIES} )
##
