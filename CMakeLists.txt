cmake_minimum_required(VERSION 2.8)
project(InvoiceUnionFind)

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

ADD_EXECUTABLE(InvoiceUnionFind main.cpp)

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  TARGET_LINK_LIBRARIES(InvoiceUnionFind ${OpenCV_LIBS})
ENDIF()
