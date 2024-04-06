find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(fmt REQUIRED)

# Define the c++ standard.
add_definitions(-std=c++17)


file(GLOB geopool_TEST_SOURCES *.cc)

foreach(geopool_TEST_SOURCE ${geopool_TEST_SOURCES})
    # Extract extension fomr file name.
    get_filename_component(geopool_TEST ${geopool_TEST_SOURCE} NAME_WE)

    add_executable(${geopool_TEST} ${geopool_TEST_SOURCE})
    target_link_libraries(${geopool_TEST} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    target_link_libraries(${geopool_TEST} fmt::fmt)
    target_link_libraries(${geopool_TEST} geopool)

    # Define a new test from the source file.
    add_test(NAME ${geopool_TEST} COMMAND ${geopool_TEST})
endforeach(geopool_TEST_SOURCE)
