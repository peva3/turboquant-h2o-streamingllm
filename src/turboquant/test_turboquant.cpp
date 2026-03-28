#include "turboquant_impl.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>

int main() {
    std::cout << "Testing TurboQuant implementation..." << std::endl;
    
    // Test parameters
    const int embedding_dim = 64;
    const int bit_width = 4; // 4-bit quantization
    const int num_vectors = 1000;
    
    // Initialize TurboQuant
    turboquant::TurboQuantKVCache tq(embedding_dim, bit_width, 42);
    
    // Generate test vectors (normal distribution)
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<std::vector<float>> test_vectors(num_vectors, std::vector<float>(embedding_dim));
    for (auto& vec : test_vectors) {
        for (float& val : vec) {
            val = dist(gen);
        }
    }
    
    // Test quantization/dequantization
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::pair<std::vector<uint8_t>, float>> quantized_results;
    std::vector<std::vector<float>> dequantized_results;
    float total_mse = 0.0f;
    
    for (const auto& vec : test_vectors) {
        auto [indices, qjl_bit, vector_norm] = tq.quantize(vec.data(), embedding_dim);
        quantized_results.push_back({indices, qjl_bit});
        
        auto dequantized = tq.dequantize(indices);
        dequantized_results.push_back(dequantized);
        
        // Compute MSE for this vector
        float mse = 0.0f;
        for (int i = 0; i < embedding_dim; ++i) {
            float diff = vec[i] - dequantized[i];
            mse += diff * diff;
        }
        total_mse += mse / embedding_dim;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float avg_mse = total_mse / num_vectors;
    
    std::cout << "Processed " << num_vectors << " vectors of dimension " << embedding_dim << std::endl;
    std::cout << "Bit width: " << bit_width << std::endl;
    std::cout << "Average MSE: " << avg_mse << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << (num_vectors * 1000.0f / duration.count()) << " vectors/sec" << std::endl;
    
    // Test inner product estimation (simplified)
    std::cout << "\nTesting inner product estimation..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    float total_ip_error = 0.0f;
    const int num_ip_tests = 100;
    
    for (int i = 0; i < num_ip_tests; ++i) {
        // Get two random vectors
        const auto& vec1 = test_vectors[i % num_vectors];
        const auto& vec2 = test_vectors[(i + 100) % num_vectors];
        
        // Compute true inner product
        float true_ip = 0.0f;
        for (int j = 0; j < embedding_dim; ++j) {
            true_ip += vec1[j] * vec2[j];
        }
        
        // Quantize both vectors
        auto [q1_indices, q1_qjl_bit, q1_norm] = tq.quantize(vec1.data(), embedding_dim);
        auto [q2_indices, q2_qjl_bit, q2_norm] = tq.quantize(vec2.data(), embedding_dim);
        
        // Estimate inner product (MSE component only in this simplified version)
        float estimated_ip = tq.estimate_inner_product(q1_indices, q1_qjl_bit, q2_indices, q2_qjl_bit);
        
        // Compute error
        float error = std::abs(true_ip - estimated_ip);
        total_ip_error += error;
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float avg_ip_error = total_ip_error / num_ip_tests;
    
    std::cout << "Processed " << num_ip_tests << " inner product estimations" << std::endl;
    std::cout << "Average IP error: " << avg_ip_error << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    
    std::cout << "\nTurboQuant test completed successfully!" << std::endl;
    return 0;
}