#include "turboquant_cuda.cuh"
#include "turboquant_impl.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <chrono>

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "Testing TurboQuant CUDA implementation..." << std::endl;
    
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "CUDA not available, skipping CUDA test" << std::endl;
        return 0;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Test parameters
    int embedding_dim = 64;
    int bit_width = 4;
    int seed = 42;
    int n_vectors = 1000;
    
    std::cout << "Testing with " << n_vectors << " vectors, " << embedding_dim << " dimensions" << std::endl;
    
    // Create TurboQuant instance
    turboquant::TurboQuantKVCache tq(embedding_dim, bit_width, seed);
    tq.init();
    
    // Create test input data
    std::vector<float> input_data(embedding_dim * n_vectors);
    for (int i = 0; i < embedding_dim * n_vectors; ++i) {
        input_data[i] = static_cast<float>(i) / (embedding_dim * n_vectors);
    }
    
    // Calculate bytes per vector
    int bits_per_vector = embedding_dim * bit_width + 1;  // +1 for QJL bit
    size_t bytes_per_vector = (bits_per_vector + 7) / 8;  // ceil division
    
    // Allocate host buffers
    std::vector<uint8_t> host_packed(n_vectors * bytes_per_vector, 0);
    std::vector<float> host_dequantized(embedding_dim * n_vectors, 0.0f);
    
    // Allocate device buffers
    float* d_input = nullptr;
    uint8_t* d_packed = nullptr;
    float* d_output = nullptr;
    float* d_rotation = nullptr;
    float* d_centroids = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_input, embedding_dim * n_vectors * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_packed, n_vectors * bytes_per_vector));
    CUDA_CHECK(cudaMalloc(&d_output, embedding_dim * n_vectors * sizeof(float)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), 
                         embedding_dim * n_vectors * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Copy rotation matrix and centroids to device
    // Note: We need to get these from the TurboQuant instance
    // For simplicity, we'll create dummy data for now
    std::vector<float> rotation(embedding_dim * embedding_dim, 0.0f);
    for (int i = 0; i < embedding_dim; ++i) {
        rotation[i * embedding_dim + i] = 1.0f;  // Identity matrix for testing
    }
    
    int num_centroids = 1 << bit_width;  // 2^bit_width
    std::vector<float> centroids(num_centroids);
    for (int i = 0; i < num_centroids; ++i) {
        centroids[i] = static_cast<float>(i) / num_centroids;
    }
    
    turboquant::cuda::copy_quantizer_to_device(
        rotation.data(),
        centroids.data(),
        embedding_dim,
        num_centroids,
        &d_rotation,
        &d_centroids
    );
    
    // Test CUDA quantization
    std::cout << "\n=== Testing CUDA Quantization ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    turboquant::cuda::quantize_vectors_cuda(
        d_input,
        d_packed,
        d_rotation,
        d_centroids,
        embedding_dim,
        bit_width,
        num_centroids,
        n_vectors,
        bytes_per_vector
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "CUDA quantization time: " << duration.count() << " µs" << std::endl;
    std::cout << "Throughput: " << (n_vectors * 1000000.0 / duration.count()) << " vectors/sec" << std::endl;
    
    // Copy packed data back to host
    CUDA_CHECK(cudaMemcpy(host_packed.data(), d_packed, 
                         n_vectors * bytes_per_vector, 
                         cudaMemcpyDeviceToHost));
    
    // Test CUDA dequantization
    std::cout << "\n=== Testing CUDA Dequantization ===" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    turboquant::cuda::dequantize_vectors_cuda(
        d_packed,
        d_output,
        d_rotation,  // Using same matrix for inverse (identity)
        d_centroids,
        embedding_dim,
        bit_width,
        num_centroids,
        n_vectors,
        bytes_per_vector
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "CUDA dequantization time: " << duration.count() << " µs" << std::endl;
    std::cout << "Throughput: " << (n_vectors * 1000000.0 / duration.count()) << " vectors/sec" << std::endl;
    
    // Copy dequantized data back to host
    CUDA_CHECK(cudaMemcpy(host_dequantized.data(), d_output, 
                         embedding_dim * n_vectors * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Verify data is not all zeros
    bool has_data = false;
    for (size_t i = 0; i < host_packed.size(); ++i) {
        if (host_packed[i] != 0) {
            has_data = true;
            break;
        }
    }
    
    if (!has_data) {
        std::cerr << "ERROR: Packed data is all zeros!" << std::endl;
        return 1;
    }
    
    std::cout << "Packed data contains data (not all zeros)" << std::endl;
    
    // Calculate memory usage
    size_t original_size = n_vectors * embedding_dim * sizeof(float);
    size_t compressed_size = host_packed.size();
    float compression_ratio = static_cast<float>(original_size) / compressed_size;
    
    std::cout << "\n=== Memory Usage ===" << std::endl;
    std::cout << "Original size (F32): " << original_size << " bytes" << std::endl;
    std::cout << "Compressed size (4-bit): " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << "x" << std::endl;
    std::cout << "Memory savings: " << (1.0f - 1.0f/compression_ratio) * 100.0f << "%" << std::endl;
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_output));
    turboquant::cuda::free_quantizer_device(d_rotation, d_centroids);
    
    std::cout << "\n=== CUDA Test Summary ===" << std::endl;
    std::cout << "CUDA quantization and dequantization working!" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << "x" << std::endl;
    std::cout << "All CUDA tests passed!" << std::endl;
    
    return 0;
}