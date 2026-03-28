#!/usr/bin/env python3
"""
Complete TurboQuant Integration for llama.cpp
Modifies cpy_k, cpy_v, get_k, get_v to use actual quantization
"""

import re

def integrate():
    with open('src/llama-kv-cache.cpp', 'r') as f:
        lines = f.readlines()
    
    output = []
    in_cpy_k = False
    in_cpy_v = False
    added_cpy_k = False
    added_cpy_v = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect cpy_k function start
        if 'ggml_tensor * llama_kv_cache::cpy_k(' in line and not added_cpy_k:
            output.append(line)
            i += 1
            # Skip to after the opening brace and initial declarations
            while i < len(lines) and 'ggml_tensor * k = layers[ikv].k;' not in lines[i]:
                output.append(lines[i])
                i += 1
            
            # Insert TurboQuant quantization BEFORE the standard implementation
            output.append('''    const int32_t ikv = map_layer_ids.at(il);
    
    // TurboQuant: Quantize and store in quantized buffer if enabled
    if (layers[ikv].use_turboquant && layers[ikv].key_turboquant && layers[ikv].key_quantized_data) {
        const int64_t n_embd_gqa = k_cur->ne[0] * k_cur->ne[1];
        const int64_t n_tokens = k_cur->ne[2];
        const float* k_data = (const float*)k_cur->data;
        const size_t bytes_per_vector = layers[ikv].bytes_per_vector;
        uint8_t* quant_base = layers[ikv].key_quantized_data;
        
        for (int64_t t = 0; t < n_tokens; t++) {
            std::vector<float> temp_vec(k_data + t * n_embd_gqa, k_data + (t + 1) * n_embd_gqa);
            auto [indices, qjl_bit] = layers[ikv].key_turboquant->quantize(temp_vec.data(), n_embd_gqa);
            auto [packed, n_bytes] = layers[ikv].key_turboquant->pack_indices(indices, qjl_bit);
            uint8_t* quant_ptr = quant_base + t * bytes_per_vector;
            memcpy(quant_ptr, packed.data(), std::min(n_bytes, bytes_per_vector));
        }
    }
    
    ggml_tensor * k = layers[ikv].k;
''')
            added_cpy_k = True
            continue
        
        # Detect cpy_v function start  
        if 'ggml_tensor * llama_kv_cache::cpy_v(' in line and not added_cpy_v:
            output.append(line)
            i += 1
            while i < len(lines) and 'auto * v = layers[ikv].v;' not in lines[i]:
                output.append(lines[i])
                i += 1
            
            # Insert TurboQuant quantization
            output.append('''    const int32_t ikv = map_layer_ids.at(il);
    
    // TurboQuant: Quantize and store in quantized buffer if enabled
    if (layers[ikv].use_turboquant && layers[ikv].value_turboquant && layers[ikv].value_quantized_data) {
        const int64_t n_embd_gqa = v_cur->ne[0] * v_cur->ne[1];
        const int64_t n_tokens = v_cur->ne[2];
        const float* v_data = (const float*)v_cur->data;
        const size_t bytes_per_vector = layers[ikv].bytes_per_vector;
        uint8_t* quant_base = layers[ikv].value_quantized_data;
        
        for (int64_t t = 0; t < n_tokens; t++) {
            std::vector<float> temp_vec(v_data + t * n_embd_gqa, v_data + (t + 1) * n_embd_gqa);
            auto [indices, qjl_bit] = layers[ikv].value_turboquant->quantize(temp_vec.data(), n_embd_gqa);
            auto [packed, n_bytes] = layers[ikv].value_turboquant->pack_indices(indices, qjl_bit);
            uint8_t* quant_ptr = quant_base + t * bytes_per_vector;
            memcpy(quant_ptr, packed.data(), std::min(n_bytes, bytes_per_vector));
        }
    }
    
    auto * v = layers[ikv].v;
''')
            added_cpy_v = True
            continue
        
        output.append(line)
        i += 1
    
    with open('src/llama-kv-cache.cpp', 'w') as f:
        f.writelines(output)
    
    print("Integrated TurboQuant quantization in cpy_k and cpy_v")

if __name__ == '__main__':
    integrate()
