
cmake_minimum_required(VERSION 2.8)
 
if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

project      (plane_sweep)
find_package (PCL REQUIRED)

SET(CMAKE_PREFIX_PATH "D:/Software/VTK 6.2.0")
find_package (VTK REQUIRED NO_MODULE)
include(${VTK_USE_FILE})

find_package(CUDA REQUIRED)

# includes from CUDA samples:
include_directories(
    "common\\inc"
    "common\\UtilNPP"
    "common\\FreeImage\\include"
    )
link_directories(
    "common\\FreeImage\\lib\\x64"
    )

set(FREEIMAGE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/common/FreeImage/lib/x64/FreeImage64.lib")

if(${VTK_VERSION} VERSION_GREATER "6" AND VTK_QT_VERSION VERSION_GREATER "4")
  # Instruct CMake to run moc automatically when needed.
  set(CMAKE_AUTOMOC ON)
  find_package(Qt5Widgets REQUIRED QUIET)
else()
  find_package(Qt4 REQUIRED)
  include(${QT_USE_FILE})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${PCL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
link_directories    (${PCL_LIBRARY_DIRS})
add_definitions     (${PCL_DEFINITIONS})

file(GLOB UI_FILES *.ui)
file(GLOB QT_WRAP *.h)
file(GLOB CXX_FILES *.cxx *.cpp)
file(GLOB CU_FILES *.cu)

if(${VTK_VERSION} VERSION_GREATER "6" AND VTK_QT_VERSION VERSION_GREATER "4")
  qt5_wrap_ui(UISrcs ${UI_FILES} )
  # CMAKE_AUTOMOC in ON so the MocHdrs will be automatically wrapped.
  cuda_add_executable(plane_sweep MACOSX_BUNDLE
    ${CXX_FILES} ${UISrcs} ${QT_WRAP} ${CU_FILES})
  qt5_use_modules(plane_sweep Core Gui)
  target_link_libraries(plane_sweep ${VTK_LIBRARIES} ${FREEIMAGE_LIB} ${PCL_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY})
else()
  QT4_WRAP_UI(UISrcs ${UI_FILES})
  QT4_WRAP_CPP(MOCSrcs ${QT_WRAP})
  cuda_add_executable(plane_sweep MACOSX_BUNDLE ${CXX_FILES} ${UISrcs} ${MOCSrcs} ${CU_FILES})

  if(VTK_LIBRARIES)
    if(${VTK_VERSION} VERSION_LESS "6")
      target_link_libraries(plane_sweep ${PCL_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} ${VTK_LIBRARIES} QVTK)
    else()
      target_link_libraries(plane_sweep ${PCL_LIBRARIES} ${CUDA_LIBRARIES} ${VTK_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY})
    endif()
  else()
    target_link_libraries(plane_sweep vtkHybrid QVTK vtkViews ${QT_LIBRARIES} ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY})
  endif()
endif()
