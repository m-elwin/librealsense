# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2022 Intel Corporation. All Rights Reserved.
cmake_minimum_required(VERSION 3.1.0)
project( pyrsutils )

set( DEPENDENCIES rsutils )

set( PYRSUTILS_FILES
    "pyrsutils.cpp"
    )

pybind11_add_module( ${PROJECT_NAME} SHARED ${PYRSUTILS_FILES} )

target_link_libraries( ${PROJECT_NAME} PRIVATE ${DEPENDENCIES} )
set_target_properties( ${PROJECT_NAME}
    PROPERTIES
        VERSION ${REALSENSE_VERSION_STRING}
        SOVERSION "${REALSENSE_VERSION_MAJOR}.${REALSENSE_VERSION_MINOR}"
    )
set_target_properties( ${PROJECT_NAME}
    PROPERTIES
        FOLDER Library/Python
    )

using_easyloggingpp( ${PROJECT_NAME} )

find_package(Python REQUIRED COMPONENTS Interpreter Development)
set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/pyrealsense2" CACHE PATH "Installation directory for Python bindings")

install(
    TARGETS ${PROJECT_NAME}
    EXPORT pyrealsense2Targets
    LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
    )
