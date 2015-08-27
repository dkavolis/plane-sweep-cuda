cmake_minimum_required(VERSION 2.8)

SET(OpenCV_DIR "D:/Software/opencv/build")
SET(VTK_DIR "D:/Software/VTK 6.2.0")

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

project      (plane_sweep)
add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# include project headers
include_directories("inc")

find_package (PCL REQUIRED)

find_package (VTK REQUIRED PATHS VTK_DIR NO_DEFAULT_PATH)
include(${VTK_USE_FILE})

find_package(CUDA REQUIRED)
add_definitions(-DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})
SET(CUDA_NVCC_FLAGS -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})

# includes from CUDA samples:
include_directories(
    "common/inc"
    ${CMAKE_CURRENT_BINARY_DIR}     # generated header files from *.ui
    )

if(${VTK_VERSION} VERSION_GREATER "6" AND VTK_QT_VERSION VERSION_GREATER "4")
  # Instruct CMake to run moc automatically when needed.
  set(CMAKE_AUTOMOC ON)
  find_package(Qt5Widgets REQUIRED QUIET)
else()
  find_package(Qt4 REQUIRED)
  include(${QT_USE_FILE})
endif()

# Project uses dynamic linking
set(OpenCV_STATIC OFF)
find_package(OpenCV PATHS OpenCV_DIR NO_DEFAULT_PATH)
if (OpenCV_FOUND)
    add_definitions(-DOpenCV_FOUND)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${PCL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS})
link_directories    (${PCL_LIBRARY_DIRS} ${OpenCV_LIB_DIR})
add_definitions     (${PCL_DEFINITIONS})

file(GLOB UI_FILES *.ui)
file(GLOB QT_WRAP *.h ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
file(GLOB CXX_FILES *.cxx *.cpp)
file(GLOB CU_FILES *.cu)

if(${VTK_VERSION} VERSION_GREATER "6" AND VTK_QT_VERSION VERSION_GREATER "4")
  qt5_wrap_ui(UISrcs ${UI_FILES} )
  # CMAKE_AUTOMOC in ON so the MocHdrs will be automatically wrapped.
  cuda_add_executable(plane_sweep MACOSX_BUNDLE
    ${CXX_FILES} ${UISrcs} ${QT_WRAP} ${CU_FILES})
  qt5_use_modules(plane_sweep Core Gui)
  target_link_libraries(plane_sweep ${VTK_LIBRARIES} ${FREEIMAGE_LIB} ${PCL_LIBRARIES} ${CUDA_LIBRARIES}
                        ${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY} ${OpenCV_LIBS})
else()
  QT4_WRAP_UI(UISrcs ${UI_FILES})
  QT4_WRAP_CPP(MOCSrcs ${QT_WRAP})
  cuda_add_executable(plane_sweep MACOSX_BUNDLE ${CXX_FILES} ${UISrcs} ${MOCSrcs} ${CU_FILES})

  if(VTK_LIBRARIES)
    if(${VTK_VERSION} VERSION_LESS "6")
      target_link_libraries(plane_sweep ${FREEIMAGE_LIB} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CUDA_LIBRARIES}
                                        ${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY} ${VTK_LIBRARIES} QVTK)
    else()
      target_link_libraries(plane_sweep ${FREEIMAGE_LIB} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CUDA_LIBRARIES}
                                        ${VTK_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY})
    endif()
  else()
    target_link_libraries(plane_sweep vtkHybrid QVTK vtkViews ${FREEIMAGE_LIB} ${OpenCV_LIBS} ${QT_LIBRARIES}
                                        ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY})
  endif()
endif()
