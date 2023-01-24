cmake_minimum_required(VERSION 3.0.2)
project(esimcpp)

find_package(
	catkin REQUIRED COMPONENTS
	pybind11_catkin
	esim
)

catkin_package(
	LIBRARIES
	CATKIN_DEPENDS pybind11_catkin
)

include_directories(${catkin_INCLUDE_DIRS})

pybind_add_module(${PROJECT_NAME} MODULE src/esim.cpp)

catkin_install_python(
	PROGRAMS scripts/esim.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
	TARGETS ${PROJECT_NAME}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
