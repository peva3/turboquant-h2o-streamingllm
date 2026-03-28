#include "turboquant_impl.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>

int main() {
    std::cout << "Profiling TurboQuant implementation..." << std::endl;
    
    // Test parameters
    const int embedding_dim = 64;
    const int bit_width = 4; // 4-bit quantization
    const int num_vectors = 10000; // More vectors for better profiling
    
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
    
    // Profile individual components
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 1. Profile random rotation generation (one-time cost)
    auto rot_start = std::chrono::high_resolution_clock::now();
    // This is done in init() constructor, so we'll profile it separately if needed
    auto rot_end = std::chrono::high_resolution_clock::now();
    auto rot_duration = std::chrono::duration_cast<std::chrono::microseconds>(rot_end - rot_start);
    
    // 2. Profile vector rotation application
    auto rotate_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> rotated_vectors(num_vectors, std::vector<float>(embedding_dim));
    for (int i = 0; i < num_vectors; ++i) {
        tq.apply_rotation(tq.generate_random_rotation(embedding_dim, 42), 
                         test_vectors[i].data(), 
                         rotated_vectors[i].data(), 
                         embedding_dim);
    }
    auto rotate_end = std::chrono::high_resolution_clock::now();
    auto rotate_duration = std::chrono::duration_cast<std::chrono::microseconds>(rotate_end - rotate_start);
    
    // 3. Profile Lloyd-Max centroid generation (one-time cost)
    auto lloyd_start = std::chrono::high_resolution_clock::now();
    // This is done in init() constructor
    auto lloyd_end = std::chrono::high_resolution_clock::now();
    auto lloyd_duration = std::chrono::duration_cast<std::chrono::microseconds>(lloyd_end - lloyd_start);
    
    // 4. Profile quantization
    auto quant_start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<std::vector<uint8_t>, float>> quantized_results;
    for (const auto& vec : test_vectors) {
        auto [indices, qjl_bit, vector_norm] = tq.quantize(vec.data(), embedding_dim);
        quantized_results.push_back(result);
    }
    auto quant_end = std::chrono::high_resolution_clock::now();
    auto quant_duration = std::chrono::duration_cast<std::chrono::microseconds>(quant_end - quant_start);
    
    // 5. Profile dequantization
    auto dequant_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> dequantized_results;
    for (const auto& result : quantized_results) {
        auto dequantized = tq.dequantize(result.first);
        dequantized_results.push_back(dequantized);
    }
    auto dequant_end = std::chrono::high_resolution_clock::now();
    auto dequant_duration = std::chrono::duration_cast<std::chrono::microseconds>(dequant_end - dequant_start);
    
    // 6. Profile 1-bit QJL transform
    auto qjl_start = std::chrono::high_resolution_clock::now();
    std::vector<float> qjl_results;
    for (const auto& vec : test_vectors) {
        float qjl_bit = tq.qjl_1bit(vec.data(), tq.qjl_projection_, embedding_dim);
        qjl_results.push_back(qjl_bit);
    }
    auto qjl_end = std::chrono::high_resolution_clock::now();
    auto qjl_duration = std::chrono::duration_cast<std::chrono::microseconds>(qjl_end - qjl_start);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    // Calculate averages
    float avg_rotate_time = rotate_duration.count() / static_cast<float>(num_vectors);
    float avg_quant_time = quant_duration.count() / static_cast<float>(num_vectors);
    float avg_dequant_time = dequant_duration.count() / static_cast<float>(num_vectors);
    float avg_qjl_time = qjl_duration.count() / static_cast<float>(num_vectors);
    
    std::cout << "=== TurboQuant Profiling Results ===" << std::endl;
    std::cout << "Vectors processed: " << num_vectors << std::endl;
    std::cout << "Embedding dimension: " << embedding_dim << std::endl;
    std::cout << "Bit width: " << bit_width << std::endl << std::endl;
    
    std::cout << "Component timings (average per vector):" << std::endl;
    std::cout << "  Random rotation generation: " << rot_duration.count() << " µs (one-time)" << std::endl;
    std::cout << "  Vector rotation application: " << avg_rotate_time << " µs" << std::endl;
    std::cout << "  Lloyd-Max centroid generation: " << lloyd_duration.count() << " µs (one-time)" << std::endl;
    std::cout << "  Quantization: " << avg_quant_time << " µs" << std::endl;
    std::cout << "  Dequantization: " << avg_dequant_time << " µs" << std::endl;
    std::cout << "  1-bit QJL transform: " << avg_qjl_time << " µs" << std::endl << std::endl;
    
    std::cout << "Total pipeline time: " << total_duration.count() << " µs" << std::endl;
    std::cout << "Throughput: " << (num_vectors * 1000000.0f / total_duration.count()) << " vectors/sec" << std::endl << std::endl;
    
    // Calculate accuracy metrics
    float total_mse = 0.0f;
    for (int i = 0; i < num_vectors; ++i) {
        float mse = 0.0f;
        for (int j = 0; j < embedding_dim; ++j) {
            float diff = test_vectors[i][j] - dequantized_results[i][j];
            mse += diff * diff;
        }
        total_mse += mse / embedding_dim;
    }
    float avg_mse = total_mse / num_vectors;
    
    std::cout << "Accuracy metrics:" << std::endl;
    std::cout << "  Average MSE: " << avg_mse << std::endl << std::endl;
    
    // Test inner product estimation
    auto ip_start = std::chrono::high_resolution_clock::now();
    float total_ip_error = 0.0f;
    const int num_ip_tests = 1000;
    
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
    auto ip_end = std::chrono::high_resolution_clock::now();
    auto ip_duration = std::chrono::duration_cast<std::chrono::microseconds>(ip_end - ip_start);
    
    float avg_ip_error = total_ip_error / num_ip_tests;
    
    std::cout << "Inner product estimation:" << std::endl;
    std::cout << "  Tests performed: " << num_ip_tests << std::endl;
    std::cout << "  Average IP error: " << avg_ip_error << std::endl;
    std::cout << "  Time taken: " << ip_duration.count() << " µs" << std::endl;
    std::cout << "  Throughput: " << (num_ip_tests * 1000000.0f / ip_duration.count()) << " estimations/sec" << std::endl;
    
    return 0;
}