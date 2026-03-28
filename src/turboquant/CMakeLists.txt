# TurboQuant KV Cache Quantization

add_library(turboquant
turboquant_impl.h
turboquant_impl.cpp turboquant_simd.h turboquant_simd.cpp
turboquant_op.h
turboquant_op.cpp
h2o_attention_accumulator.h
h2o_attention_accumulator.cpp turboquant_cuda_streams.h turboquant_cuda_streams.cpp
)

target_include_directories(turboquant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

if (GGML_CUDA)
# Add CUDA implementation
enable_language(CUDA)

# Add CUDA files to the library
target_sources(turboquant PRIVATE turboquant_cuda.cu)

# Set CUDA properties and architecture
set_target_properties(turboquant PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_ARCHITECTURES "89" # RTX 4060 Ti compute capability
)

# Add CUDA include directories
target_include_directories(turboquant PRIVATE
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

message(STATUS "TurboQuant: Building with CUDA support (architecture 89)")
elseif (GGML_HIP)
# Add AMD HIP implementation
enable_language(HIP)

# Add HIP files to the library
target_sources(turboquant PRIVATE turboquant_hip.cpp)

# Set HIP properties
set_target_properties(turboquant PROPERTIES
HIP_SEPARABLE_COMPILATION ON
HIP_RESOLVE_DEVICE_SYMBOLS ON
)

# Add HIP include directories
target_include_directories(turboquant PRIVATE
${CMAKE_HIP_INCLUDE_DIRS}
)

message(STATUS "TurboQuant: Building with HIP/AMD support")
else()
message(STATUS "TurboQuant: Building CPU-only version (CUDA/HIP not available)")
endif()

# Add test executable
add_executable(test-turboquant test_turboquant.cpp)
target_link_libraries(test-turboquant turboquant ggml-base ggml-cpu ggml)

# Add profiling executable
add_executable(test-turboquant-profiling test_turboquant_profiling.cpp)
target_link_libraries(test-turboquant-profiling turboquant ggml-base ggml-cpu ggml)

# Add packing test executable
add_executable(test-turboquant-packing test_packing.cpp)
target_link_libraries(test-turboquant-packing turboquant ggml-base ggml-cpu ggml)

# Add buffer write test executable
add_executable(test-turboquant-buffer-write test_buffer_write.cpp)
target_link_libraries(test-turboquant-buffer-write turboquant ggml-base ggml-cpu ggml)

# Add dequantization test executable
add_executable(test-turboquant-dequantization test_dequantization.cpp)
target_link_libraries(test-turboquant-dequantization turboquant ggml-base ggml-cpu ggml)

# Only build tests if explicitly requested
if (BUILD_TESTS)
    add_test(NAME turboquant-test COMMAND test-turboquant)
    add_test(NAME turboquant-profiling-test COMMAND test-turboquant-profiling)
    add_test(NAME turboquant-packing-test COMMAND test-turboquant-packing)
    add_test(NAME turboquant-buffer-write-test COMMAND test-turboquant-buffer-write)
    add_test(NAME turboquant-dequantization-test COMMAND test-turboquant-dequantization)
endif()

# Add CUDA test executable if CUDA is available
if (GGML_CUDA)
    enable_language(CUDA)
    
    add_executable(test-turboquant-cuda test_turboquant_cuda.cu)
    target_link_libraries(test-turboquant-cuda turboquant ggml-base ggml-cpu ggml)
    
    set_target_properties(test-turboquant-cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "89"  # RTX 4060 Ti compute capability
    )
    
    if (BUILD_TESTS)
        add_test(NAME turboquant-cuda-test COMMAND test-turboquant-cuda)
    endif()
endif()

# SIMD optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable AVX2 for SIMD optimizations
    if(NOT MSVC)
        add_compile_options(-mavx2 -mfma)
    endif()
endif()
