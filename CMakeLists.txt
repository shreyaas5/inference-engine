cmake_minimum_required(VERSION 3.16)

project(inference-engine
    VERSION 0.1.0
    DESCRIPTION "Custom C++17 CPU inference engine for TinyLlama-1.1B-Chat-v1.0"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(INFERENCE_ENGINE_BUILD_TESTS "Build inference-engine unit tests" ON)
option(INFERENCE_ENGINE_ENABLE_AVX2 "Enable AVX2/FMA compilation on x86_64 targets" ON)

find_package(Threads REQUIRED)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SENTENCEPIECE QUIET IMPORTED_TARGET sentencepiece>=0.2.0)
endif()

if(TARGET PkgConfig::SENTENCEPIECE)
    set(INFERENCE_ENGINE_SENTENCEPIECE_TARGET PkgConfig::SENTENCEPIECE)
else()
    include(FetchContent)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
    set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG 17d7580d6407802f85855d2cc9190634e2c95624
        GIT_SHALLOW FALSE
    )
    FetchContent_MakeAvailable(sentencepiece)
    set(INFERENCE_ENGINE_SENTENCEPIECE_TARGET sentencepiece-static)
    if(TARGET sentencepiece-static AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(sentencepiece-static PRIVATE -include cstdint)
    endif()
endif()

set(INFERENCE_ENGINE_CORE_SOURCES
    src/common/common.cpp
    src/common/tensor.cpp
    src/loader/loader.cpp
    src/core/kv_cache.cpp
    src/transformer/transformer.cpp
    src/kernels/quant_q4_0.cpp
    src/kernels/scalar/kernels_scalar.cpp
    src/kernels/neon/kernels_neon.cpp
    src/kernels/avx2/kernels_avx2.cpp
    src/kernels/threading/thread_pool.cpp
    src/kernels/dispatch.cpp
    src/tokenizer/tokenizer.cpp
    src/sampler/sampler.cpp
)

add_library(inference_engine_core STATIC ${INFERENCE_ENGINE_CORE_SOURCES})

target_include_directories(inference_engine_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(inference_engine_core
    PUBLIC
        Threads::Threads
    PRIVATE
        ${INFERENCE_ENGINE_SENTENCEPIECE_TARGET}
)

if(NOT TARGET PkgConfig::SENTENCEPIECE)
    target_include_directories(inference_engine_core SYSTEM PRIVATE
        ${sentencepiece_SOURCE_DIR}/src
        ${sentencepiece_SOURCE_DIR}/src/builtin_pb
        ${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite
        ${sentencepiece_BINARY_DIR}/src
        ${sentencepiece_BINARY_DIR}
    )
endif()

if(MSVC)
    target_compile_options(inference_engine_core PRIVATE /W4)
else()
    target_compile_options(inference_engine_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Enable AVX2 + FMA flags specifically on the AVX2 kernel translation unit on x86_64.
if(INFERENCE_ENGINE_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
    if(MSVC)
        set_source_files_properties(src/kernels/avx2/kernels_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else()
        set_source_files_properties(src/kernels/avx2/kernels_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    endif()
endif()

add_executable(inference-engine src/app/main.cpp)
target_link_libraries(inference-engine PRIVATE inference_engine_core)
if(WIN32)
    target_link_libraries(inference-engine PRIVATE Psapi)
endif()

if(INFERENCE_ENGINE_BUILD_TESTS)
    enable_testing()

    add_executable(inference_engine_tests
        tests/test_main.cpp
        tests/test_loader_validation.cpp
        tests/test_quant_packing.cpp
        tests/test_kernels_scalar.cpp
        tests/test_kernels_simd.cpp
        tests/test_kv_cache.cpp
        tests/test_tokenizer.cpp
        tests/test_sampler.cpp
        tests/test_golden_forward.cpp
    )

    target_link_libraries(inference_engine_tests PRIVATE inference_engine_core)

    if(MSVC)
        target_compile_options(inference_engine_tests PRIVATE /W4)
    else()
        target_compile_options(inference_engine_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    set(INFERENCE_ENGINE_TEST_SUITES
        loader_validation
        quant_packing
        kernels_scalar
        kernels_simd
        kv_cache
        tokenizer
        sampler
        golden_forward
    )

    foreach(SUITE_NAME IN LISTS INFERENCE_ENGINE_TEST_SUITES)
        add_test(
            NAME ${SUITE_NAME}
            COMMAND inference_engine_tests --suite ${SUITE_NAME}
        )
    endforeach()

    set_tests_properties(tokenizer PROPERTIES
        ENVIRONMENT "INFERENCE_ENGINE_TOKENIZER_MODEL=${CMAKE_CURRENT_BINARY_DIR}/tokenizer/tokenizer.model"
    )
endif()
