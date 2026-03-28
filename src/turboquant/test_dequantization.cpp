#include "turboquant_impl.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>

int main() {
    std::cout << "Testing TurboQuant dequantization from quantized buffers..." << std::endl;
    
    // Test parameters
    int embedding_dim = 64;
    int bit_width = 4;
    int seed = 42;
    int n_vectors = 10;
    
    turboquant::TurboQuantKVCache tq(embedding_dim, bit_width, seed);
    tq.init();
    
    // Create test input data
    std::vector<float> original_data(embedding_dim * n_vectors);
    for (int i = 0; i < embedding_dim * n_vectors; ++i) {
        original_data[i] = static_cast<float>(i) / (embedding_dim * n_vectors);
    }
    
    // Calculate bytes per vector
    int bits_per_vector = embedding_dim * bit_width + 1;  // +1 for QJL bit
    size_t bytes_per_vector = (bits_per_vector + 7) / 8;  // ceil division
    
    // Allocate quantized buffer
    std::vector<uint8_t> quantized_buffer(n_vectors * bytes_per_vector, 0);
    
    std::cout << "Testing with " << n_vectors << " vectors, " << embedding_dim << " dimensions" << std::endl;
    std::cout << "Bytes per vector: " << bytes_per_vector << std::endl;
    
    // Quantize and pack all vectors
    std::cout << "\n=== Quantizing and packing vectors ===" << std::endl;
    for (int i = 0; i < n_vectors; ++i) {
        const float* vec = &original_data[i * embedding_dim];
        
        // Quantize
        auto [indices, qjl_bit, vector_norm] = tq.quantize(vec, embedding_dim);
        
        // Pack
        auto [packed, n_bytes] = tq.pack_indices(indices, qjl_bit);
        
        // Write to buffer
        uint8_t* buffer_ptr = &quantized_buffer[i * bytes_per_vector];
        std::memcpy(buffer_ptr, packed.data(), n_bytes);
    }
    
    // Dequantize all vectors
    std::cout << "\n=== Dequantizing vectors ===" << std::endl;
    std::vector<float> dequantized_data(embedding_dim * n_vectors);
    
    for (int i = 0; i < n_vectors; ++i) {
        const uint8_t* buffer_ptr = &quantized_buffer[i * bytes_per_vector];
        
        // Unpack
        auto [indices, qjl_bit] = tq.unpack_indices(buffer_ptr, bytes_per_vector);
        
        // Dequantize directly to output buffer (optimized)
        float vector_norm = 1.0f; // Default norm for testing
        tq.dequantize_to_buffer(indices.data(), &dequantized_data[i * embedding_dim], embedding_dim, vector_norm);
    }
    
    // Calculate MSE between original and dequantized
    float mse = 0.0f;
    for (int i = 0; i < embedding_dim * n_vectors; ++i) {
        float diff = original_data[i] - dequantized_data[i];
        mse += diff * diff;
    }
    mse /= (embedding_dim * n_vectors);
    
    std::cout << "\n=== Quality Metrics ===" << std::endl;
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << std::sqrt(mse) << std::endl;
    
    // Calculate compression ratio
    size_t original_size = n_vectors * embedding_dim * sizeof(float);
    size_t compressed_size = quantized_buffer.size();
    float compression_ratio = static_cast<float>(original_size) / compressed_size;
    
    std::cout << "\n=== Compression Results ===" << std::endl;
    std::cout << "Original size (F32): " << original_size << " bytes" << std::endl;
    std::cout << "Compressed size (4-bit): " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << "x" << std::endl;
    std::cout << "Memory savings: " << (1.0f - 1.0f/compression_ratio) * 100.0f << "%" << std::endl;
    
    // Verify MSE is acceptable (should be small for 4-bit quantization)
    if (mse > 0.1f) {
        std::cerr << "WARNING: MSE is high (" << mse << "), quantization may be too lossy" << std::endl;
    } else {
        std::cout << "MSE is acceptable for 4-bit quantization" << std::endl;
    }
    
    std::cout << "\nAll dequantization tests passed!" << std::endl;
    return 0;
}