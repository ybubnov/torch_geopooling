find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(fmt REQUIRED)
find_package(torch REQUIRED)

# Define the c++ standard.
add_definitions(-std=c++17)


file(GLOB torch_geopooling_TEST_SOURCES *.cc)

foreach(torch_geopooling_TEST_SOURCE ${torch_geopooling_TEST_SOURCES})
    # Extract extension fomr file name.
    get_filename_component(torch_geopooling_TEST ${torch_geopooling_TEST_SOURCE} NAME_WE)

    add_executable(${torch_geopooling_TEST} ${torch_geopooling_TEST_SOURCE})
    target_link_libraries(${torch_geopooling_TEST} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    target_link_libraries(${torch_geopooling_TEST} fmt::fmt)
    target_link_libraries(${torch_geopooling_TEST} torch)
    target_link_libraries(${torch_geopooling_TEST} torch_geopooling)

    # Define a new test from the source file.
    add_test(NAME ${torch_geopooling_TEST} COMMAND ${torch_geopooling_TEST})
endforeach(torch_geopooling_TEST_SOURCE)
