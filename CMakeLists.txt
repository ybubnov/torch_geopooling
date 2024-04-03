cmake_minimum_required(VERSION 3.15)

project(gepool, CXX)
add_definitions(-std=c++17)
find_package(fmt REQUIRED)

enable_testing()

set(geopool_LIBRARY_VERSION "1.0.0")
set(geopool_LIBRARY_SOVERSION "1")


include_directories(include)
include_directories(include/geopool)

aux_source_directory(src geopool_SOURCES)

# Add the testing directory.
add_subdirectory(test)


add_library(geopool SHARED ${geopool_SOURCES})
set_target_properties(geopool PROPERTIES VERSION ${geopool_LIBRARY_VERSION})
set_target_properties(geopool PROPERTIES SOVERSION ${geopool_LIBRARY_SOVERSION})
target_link_libraries(geopool fmt::fmt)


# Install the library files.
install(DIRECTORY include/geopool DESTINATION geopool)
install(FILES include/geopool.h DESTINATION include)
install(TARGETS geopool DESTINATION lib)
