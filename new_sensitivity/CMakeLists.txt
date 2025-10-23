cmake_minimum_required(VERSION 3.16)
project(new_sensitivity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Root paths
set(KAIZEN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../kaizen/src")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${KAIZEN_SRC_DIR}
)

set(KAIZEN_SOURCES
    ${KAIZEN_SRC_DIR}/circuit.cpp
    ${KAIZEN_SRC_DIR}/elliptic_curves.cpp
    ${KAIZEN_SRC_DIR}/expander.cpp
    ${KAIZEN_SRC_DIR}/fieldElement.cpp
    ${KAIZEN_SRC_DIR}/GKR.cpp
    ${KAIZEN_SRC_DIR}/linear_code_encode.cpp
    ${KAIZEN_SRC_DIR}/lookups.cpp
    ${KAIZEN_SRC_DIR}/merkle_tree.cpp
    ${KAIZEN_SRC_DIR}/mimc.cpp
    ${KAIZEN_SRC_DIR}/polynomial.cpp
    ${KAIZEN_SRC_DIR}/pol_verifier.cpp
    ${KAIZEN_SRC_DIR}/poly_commit.cpp
    ${KAIZEN_SRC_DIR}/prover.cpp
    ${KAIZEN_SRC_DIR}/quantization.cpp
    ${KAIZEN_SRC_DIR}/utils.cpp
    ${KAIZEN_SRC_DIR}/verifier.cpp
)

file(GLOB NEW_SENSITIVITY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(GRADIENT_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM NEW_SENSITIVITY_SOURCES ${GRADIENT_MAIN})

add_library(kaizen STATIC ${KAIZEN_SOURCES})
target_compile_options(kaizen PRIVATE -mavx2)

find_library(GMP_LIB gmp PATHS ENV LIBRARY_PATH PATH_SUFFIXES lib)
set(XKCP_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../sensitivity_zkp/lib/libXKCP.a")

if (NOT EXISTS ${XKCP_LIB})
    message(FATAL_ERROR "Required third-party library libXKCP.a not found")
endif ()

target_link_libraries(kaizen PUBLIC ${GMP_LIB} ${XKCP_LIB})

add_library(gradient_core STATIC ${NEW_SENSITIVITY_SOURCES})
target_link_libraries(gradient_core PUBLIC kaizen)

add_executable(gradient_prover ${GRADIENT_MAIN})
target_link_libraries(gradient_prover PRIVATE gradient_core)

add_executable(random_dense_test tests/random_dense.cpp)
target_link_libraries(random_dense_test PRIVATE gradient_core)

add_executable(random_cnn_test tests/random_cnn.cpp)
target_link_libraries(random_cnn_test PRIVATE gradient_core)

add_executable(model_loader tests/model_loader.cpp)
target_link_libraries(model_loader PRIVATE gradient_core)
