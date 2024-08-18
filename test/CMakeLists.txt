find_package(Boost COMPONENTS unit_test_framework REQUIRED)

# Define the c++ standard.
add_definitions(-std=c++17)

include_directories()


file(GLOB torch_geopooling_TEST_SOURCES *.cc)

foreach(torch_geopooling_TEST_SOURCE ${torch_geopooling_TEST_SOURCES})
    # Extract extension from the file name.
    get_filename_component(torch_geopooling_TEST ${torch_geopooling_TEST_SOURCE} NAME_WE)

    add_executable(${torch_geopooling_TEST} ${torch_geopooling_TEST_SOURCE})
    target_link_libraries(${torch_geopooling_TEST} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    target_link_libraries(${torch_geopooling_TEST} fmt::fmt-header-only)
    target_link_libraries(${torch_geopooling_TEST} "${TORCH_LIBRARIES}")
    target_link_libraries(${torch_geopooling_TEST} torch_geopooling)

    # Define a new test from the source file.
    add_test(NAME ${torch_geopooling_TEST} COMMAND ${torch_geopooling_TEST})
endforeach(torch_geopooling_TEST_SOURCE)
