#include "turboquant_impl.h"
#include "turboquant_op.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

int main() {
    std::cout << "Testing TurboQuant custom op with quantized buffer..." << std::endl;
    
    // Test parameters
    int embedding_dim = 64;
    int bit_width = 4;
    int seed = 42;
    int n_vectors = 10;
    
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
    
    // Allocate quantized buffer
    std::vector<uint8_t> quantized_buffer(n_vectors * bytes_per_vector, 0);
    
    std::cout << "Testing with " << n_vectors << " vectors, " << embedding_dim << " dimensions" << std::endl;
    std::cout << "Bytes per vector: " << bytes_per_vector << std::endl;
    std::cout << "Total buffer size: " << quantized_buffer.size() << " bytes" << std::endl;
    
    // Test packing/unpacking directly first
    std::cout << "\n=== Testing packing/unpacking directly ===" << std::endl;
    for (int i = 0; i < n_vectors; ++i) {
        const float* vec = &input_data[i * embedding_dim];
        
        // Quantize
        auto [indices, qjl_bit, vector_norm] = tq.quantize(vec, embedding_dim);
        
        // Pack
        auto [packed, n_bytes] = tq.pack_indices(indices, qjl_bit);
        
        // Write to buffer
        uint8_t* buffer_ptr = &quantized_buffer[i * bytes_per_vector];
        std::memcpy(buffer_ptr, packed.data(), n_bytes);
        
        // Unpack and verify
        auto [unpacked_indices, unpacked_qjl] = tq.unpack_indices(buffer_ptr, n_bytes);
        
        // Verify indices match
        bool indices_match = true;
        for (size_t j = 0; j < indices.size(); ++j) {
            if (unpacked_indices[j] != indices[j]) {
                indices_match = false;
                break;
            }
        }
        
        if (!indices_match) {
            std::cerr << "ERROR: Indices mismatch for vector " << i << std::endl;
            return 1;
        }
        
        // Verify QJL bit matches (should be -1.0f or 1.0f)
        if (std::abs(unpacked_qjl - qjl_bit) > 0.1f) {
            std::cerr << "ERROR: QJL bit mismatch for vector " << i 
                      << ": " << unpacked_qjl << " vs " << qjl_bit << std::endl;
            return 1;
        }
    }
    
    std::cout << "All " << n_vectors << " vectors passed packing/unpacking test!" << std::endl;
    
    // Verify buffer is not all zeros
    bool buffer_has_data = false;
    for (size_t i = 0; i < quantized_buffer.size(); ++i) {
        if (quantized_buffer[i] != 0) {
            buffer_has_data = true;
            break;
        }
    }
    
    if (!buffer_has_data) {
        std::cerr << "ERROR: Quantized buffer is all zeros!" << std::endl;
        return 1;
    }
    
    std::cout << "Quantized buffer contains data (not all zeros)" << std::endl;
    
    // Calculate and display compression ratio
    size_t original_size = n_vectors * embedding_dim * sizeof(float);
    size_t compressed_size = quantized_buffer.size();
    float compression_ratio = static_cast<float>(original_size) / compressed_size;
    
    std::cout << "\n=== Compression Results ===" << std::endl;
    std::cout << "Original size (F32): " << original_size << " bytes" << std::endl;
    std::cout << "Compressed size (4-bit): " << compressed_size << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compression_ratio << "x" << std::endl;
    std::cout << "Memory savings: " << (1.0f - 1.0f/compression_ratio) * 100.0f << "%" << std::endl;
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}