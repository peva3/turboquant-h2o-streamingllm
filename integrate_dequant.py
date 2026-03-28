#!/usr/bin/env python3
"""
Add dequantization to get_k and get_v
"""

def integrate():
    with open('src/llama-kv-cache.cpp', 'r') as f:
        content = f.read()
    
    # Add dequantization to get_k
    get_k_old = '''ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;'''
    
    get_k_new = '''ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    
    // TurboQuant: Dequantize from quantized buffer if enabled
    if (layers[ikv].use_turboquant && layers[ikv].key_quantized_data && layers[ikv].key_turboquant) {
        const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
        const size_t buffer_offset = sinfo.s0;
        const size_t bytes_per_vector = layers[ikv].bytes_per_vector;
        const int64_t vector_size = hparams.n_embd_head_k(il) * hparams.n_head_kv(il);
        
        ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns);
        
        const uint8_t* quant_base = layers[ikv].key_quantized_data;
        float* output_ptr = (float*)result->data;
        
        for (int64_t i = 0; i < (int64_t)n_kv * ns; i++) {
            const uint8_t* qdata = quant_base + (buffer_offset + i) * bytes_per_vector;
            layers[ikv].key_turboquant->dequantize_to_buffer(qdata, output_ptr + i * vector_size, vector_size);
        }
        
        return result;
    }

    auto * k = layers[ikv].k;'''
    
    content = content.replace(get_k_old, get_k_new)
    
    # Add dequantization to get_v
    get_v_old = '''ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;'''
    
    get_v_new = '''ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    
    // TurboQuant: Dequantize from quantized buffer if enabled
    if (layers[ikv].use_turboquant && layers[ikv].value_quantized_data && layers[ikv].value_turboquant) {
        const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
        const size_t buffer_offset = sinfo.s0;
        const size_t bytes_per_vector = layers[ikv].bytes_per_vector;
        const int64_t vector_size = hparams.n_embd_head_v(il) * hparams.n_head_kv(il);
        
        ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns);
        
        const uint8_t* quant_base = layers[ikv].value_quantized_data;
        float* output_ptr = (float*)result->data;
        
        for (int64_t i = 0; i < (int64_t)n_kv * ns; i++) {
            const uint8_t* qdata = quant_base + (buffer_offset + i) * bytes_per_vector;
            layers[ikv].value_turboquant->dequantize_to_buffer(qdata, output_ptr + i * vector_size, vector_size);
        }
        
        return result;
    }

    auto * v = layers[ikv].v;'''
    
    content = content.replace(get_v_old, get_v_new)
    
    with open('src/llama-kv-cache.cpp', 'w') as f:
        f.write(content)
    
    print("Integrated dequantization in get_k and get_v")

if __name__ == '__main__':
    integrate()
