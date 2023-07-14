cmake_minimum_required( VERSION 3.1 )

# Splution(*.sln) and Project(*.vcxproj)
project( Hough )

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} hough.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})