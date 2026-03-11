cmake_minimum_required(VERSION 3.14)
project(MLSysProject CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ABSL_PROPAGATE_CXX_STD ON)

include(FetchContent)

# Fetch Abseil
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20260107.1
)
FetchContent_MakeAvailable(absl)

# Fetch nlohmann/json
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

# Fetch GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)
enable_testing()

# Define a common library for shared logic
add_library(mlsys_lib STATIC mlsys.cpp solver.cpp)
target_include_directories(mlsys_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mlsys_lib PUBLIC
    absl::status
    absl::statusor
    absl::time
    nlohmann_json::nlohmann_json
)

# Define the main executable
add_executable(mlsys main.cpp)
target_link_libraries(mlsys PRIVATE mlsys_lib)

# Define the evaluate_test executable
add_executable(evaluate_test evaluate_test.cpp)
target_link_libraries(evaluate_test PRIVATE mlsys_lib GTest::gtest_main)
target_compile_definitions(evaluate_test PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
include(GoogleTest)
gtest_discover_tests(evaluate_test)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(mlsys_lib PRIVATE DEBUG)
    target_compile_definitions(mlsys PRIVATE DEBUG)
    target_compile_definitions(evaluate_test PRIVATE DEBUG)
endif()