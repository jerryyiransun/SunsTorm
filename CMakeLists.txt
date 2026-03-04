cmake_minimum_required(VERSION 3.14)
project(MLSysProject CXX)

set(CMAKE_CXX_STANDARD 17)
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

# Define executable and its source files
add_executable(mlsys main.cpp mlsys.cpp)

# Includes files in the current source directory
target_include_directories(mlsys PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(mlsys PRIVATE
    absl::status
    absl::statusor
    nlohmann_json::nlohmann_json
)