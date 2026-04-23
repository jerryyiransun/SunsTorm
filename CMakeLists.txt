cmake_minimum_required(VERSION 3.14)
project(MLSysProject CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ABSL_PROPAGATE_CXX_STD ON)

option(MLSYS_ENABLE_FUSER_LOGGING "Enable compile-time greedy fuser logging instrumentation" OFF)

find_program(CLANG_FORMAT_EXE NAMES "clang-format")

include(FetchContent)
find_package(Threads REQUIRED)

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
add_library(mlsys_lib STATIC
    mlsys.cpp
    fuser.cpp
    fuser_logging.cpp
    tiler.cpp
    cost_model.cpp
    solver.cpp
    solution_writer.cpp
)
target_include_directories(mlsys_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(mlsys_lib PUBLIC
    MLSYS_ENABLE_FUSER_LOGGING=$<BOOL:${MLSYS_ENABLE_FUSER_LOGGING}>
)
target_link_libraries(mlsys_lib PUBLIC
    absl::status
    absl::statusor
    absl::time
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Define the main executable
add_executable(mlsys main.cpp)
target_link_libraries(mlsys PRIVATE mlsys_lib)

# Local runner with solver selection + optional logging flags
add_executable(run_solver run_solver_main.cpp)
target_link_libraries(run_solver PRIVATE mlsys_lib)

# Debug-focused evaluator runner
add_executable(evaluate_debug evaluate_debug.cpp)
target_link_libraries(evaluate_debug PRIVATE mlsys_lib)

include(GoogleTest)

function(add_mlsys_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE mlsys_lib GTest::gtest_main)
    target_compile_definitions(${target} PRIVATE
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
    )
    gtest_discover_tests(${target})
endfunction()

add_mlsys_test(evaluate_test evaluate_test.cpp)
add_mlsys_test(tiler_test tiler_test.cpp)
add_mlsys_test(fuser_test fuser_test.cpp)
add_mlsys_test(cost_model_test cost_model_test.cpp)
add_mlsys_test(solver_test solver_test.cpp)
add_mlsys_test(solution_writer_test solution_writer_test.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(mlsys_lib PRIVATE DEBUG)
    target_compile_definitions(mlsys PRIVATE DEBUG)
    target_compile_definitions(run_solver PRIVATE DEBUG)
    target_compile_definitions(evaluate_debug PRIVATE DEBUG)
    target_compile_definitions(evaluate_test PRIVATE DEBUG)
    target_compile_definitions(tiler_test PRIVATE DEBUG)
    target_compile_definitions(fuser_test PRIVATE DEBUG)
    target_compile_definitions(cost_model_test PRIVATE DEBUG)
    target_compile_definitions(solver_test PRIVATE DEBUG)
    target_compile_definitions(solution_writer_test PRIVATE DEBUG)
endif()

if(CLANG_FORMAT_EXE)
# 1. Grab every single .cpp, .h, and .hpp file in your entire project folder tree
    file(GLOB_RECURSE ALL_CXX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    )

    # 2. Filter out the _deps and build folders so we don't accidentally format Abseil/JSON
    set(PROJECT_SOURCES "")
    foreach(SOURCE_FILE ${ALL_CXX_SOURCES})
        # If the file path does NOT contain "/_deps/" and does NOT contain "/build", keep it
        if(NOT SOURCE_FILE MATCHES "/_deps/" AND NOT SOURCE_FILE MATCHES "/build")
            list(APPEND PROJECT_SOURCES ${SOURCE_FILE})
        endif()
    endforeach()

    # 3. Feed that safe, filtered list into clang-format
    if(PROJECT_SOURCES)
        add_custom_target(
            format
            COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${PROJECT_SOURCES}
            COMMENT "Running clang-format on all project files..."
        )
    else()
        message(WARNING "No source files found for clang-format.")
    endif()
endif()
