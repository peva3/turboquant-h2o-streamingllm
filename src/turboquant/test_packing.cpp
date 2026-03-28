#include "turboquant_impl.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

int main() {
    std::cout << "Testing TurboQuant packing/unpacking functions..." << std::endl;
    
    // Test parameters
    int embedding_dim = 64;
    int bit_width = 4;
    int seed = 42;
    
    turboquant::TurboQuantKVCache tq(embedding_dim, bit_width, seed);
    tq.init();
    
    // Create test data
    std::vector<float> test_vector(embedding_dim);
    for (int i = 0; i < embedding_dim; ++i) {
        test_vector[i] = static_cast<float>(i) / embedding_dim;
    }
    
    // Quantize
        auto [indices, qjl_bit, vector_norm] = tq.quantize(test_vector.data(), embedding_dim);
    
    std::cout << "Quantized " << embedding_dim << " dimensions" << std::endl;
    std::cout << "Indices size: " << indices.size() << std::endl;
    std::cout << "QJL bit: " << qjl_bit << std::endl;
    
    // Pack
    auto [packed, n_bytes] = tq.pack_indices(indices, qjl_bit);
    
    std::cout << "Packed size: " << n_bytes << " bytes" << std::endl;
    std::cout << "Expected size: " << (embedding_dim * bit_width + 7) / 8 << " bytes" << std::endl;
    
    // Verify packing size
    int expected_bytes = (embedding_dim * bit_width + 1 + 7) / 8; // +1 for QJL bit
    std::cout << "Expected bytes calculation: (" << embedding_dim << " * " << bit_width << " + 1 + 7) / 8 = " << expected_bytes << std::endl;
    assert(n_bytes == expected_bytes);
    
    // Unpack
    auto [unpacked_indices, unpacked_qjl] = tq.unpack_indices(packed.data(), n_bytes);
    
    // Verify unpacked data matches original
    assert(unpacked_indices.size() == indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        if (unpacked_indices[i] != indices[i]) {
            std::cerr << "Mismatch at index " << i << ": " 
                      << (int)unpacked_indices[i] << " vs " << (int)indices[i] << std::endl;
            return 1;
        }
    }
    
    // Verify QJL bit matches
    if (std::abs(unpacked_qjl - qjl_bit) > 0.1f) {
        std::cerr << "QJL bit mismatch: " << unpacked_qjl << " vs " << qjl_bit << std::endl;
        return 1;
    }
    
    // Test dequantization
    auto dequantized = tq.dequantize(unpacked_indices);
    assert(dequantized.size() == embedding_dim);
    
    // Calculate MSE
    float mse = 0.0f;
    for (int i = 0; i < embedding_dim; ++i) {
        float diff = test_vector[i] - dequantized[i];
        mse += diff * diff;
    }
    mse /= embedding_dim;
    
    std::cout << "Quantization MSE: " << mse << std::endl;
    std::cout << "All tests passed!" << std::endl;
    
    // Test memory savings
    size_t original_size = embedding_dim * sizeof(float); // F32
    size_t compressed_size = n_bytes;
    float compression_ratio = static_cast<float>(original_size) / compressed_size;
    
    std::cout << "\nMemory savings:" << std::endl;
    std::cout << "Original (F32): " << original_size << " bytes" << std::endl;
    std::cout << "TurboQuant (4-bit): " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << "x" << std::endl;
    
    return 0;
}