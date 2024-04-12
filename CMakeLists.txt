cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(torch_geopooling, CXX)
add_definitions(-std=c++17)

find_package(fmt REQUIRED)
find_package(Torch REQUIRED)

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(torch_geopooling_LIBRARY_VERSION "1.0.0")
set(torch_geopooling_LIBRARY_SOVERSION "1")


include_directories(include)
include_directories(include/torch_geopooling)

aux_source_directory(src torch_geopooling_SOURCES)

# Add the testing directory.
add_subdirectory(test)


add_library(torch_geopooling SHARED ${torch_geopooling_SOURCES})
set_target_properties(torch_geopooling PROPERTIES CXX_STANDARD 17)
set_target_properties(torch_geopooling PROPERTIES VERSION ${torch_geopooling_LIBRARY_VERSION})
set_target_properties(torch_geopooling PROPERTIES SOVERSION ${torch_geopooling_LIBRARY_SOVERSION})
target_link_libraries(torch_geopooling fmt::fmt)
target_link_libraries(torch_geopooling "${TORCH_LIBRARIES}")


# Install the library files.
install(DIRECTORY include/torch_geopooling DESTINATION torch_geopooling)
install(FILES include/torch_geopooling.h DESTINATION include)
install(TARGETS torch_geopooling DESTINATION lib)
