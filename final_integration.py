#!/usr/bin/env python3
"""
Final complete TurboQuant integration - single comprehensive patch
"""

with open('src/llama-kv-cache.cpp', 'r') as f:
    content = f.read()

# 1. Add constructor parameter
content = content.replace('const layer_reuse_cb & reuse) :', 
                          'const layer_reuse_cb & reuse,\nbool use_turboquant) :')

# 2. Replace entire cpy_k function
cpy_k_full = '''ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);
    const int32_t ikv = map_layer_ids.at(il);
    
    // TurboQuant: Quantize if enabled
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
    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head = k_cur->ne[1];
    const int64_t n_tokens = k_cur->ne[2];
    const int64_t n_embd_gqa = n_embd_head*n_head;
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);
    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);
    const int64_t n_stream = k->ne[2];
    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size == k->ne[1]);
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}'''

# Find and replace cpy_k
import re
cpy_k_pattern = r'ggml_tensor \* llama_kv_cache::cpy_k\(ggml_context \* ctx, ggml_tensor \* k_cur, ggml_tensor \* k_idxs, int32_t il, const slot_info & sinfo\) const \{[^}]+return ggml_set_rows\(ctx, k, k_cur, k_idxs\);\n\}'
content = re.sub(cpy_k_pattern, cpy_k_full, content)

# 3. Replace entire cpy_v function  
cpy_v_full = '''ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);
    const int32_t ikv = map_layer_ids.at(il);
    
    // TurboQuant: Quantize if enabled
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
    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head = v_cur->ne[1];
    const int64_t n_tokens = v_cur->ne[2];
    const int64_t n_embd_gqa = n_embd_head*n_head;
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);
    v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);
    const int64_t n_stream = v->ne[2];
    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        assert(n_embd_gqa == v->ne[0]);
        assert(kv_size == v->ne[1]);
        v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
    }
    return ggml_set_rows(ctx, v, v_cur, v_idxs);
}'''

cpy_v_pattern = r'ggml_tensor \* llama_kv_cache::cpy_v\(ggml_context \* ctx, ggml_tensor \* v_cur, ggml_tensor \* v_idxs, int32_t il, const slot_info & sinfo\) const \{[^}]+return ggml_set_rows\(ctx, v, v_cur, v_idxs\);\n\}'
content = re.sub(cpy_v_pattern, cpy_v_full, content)

# 4. Replace get_k with dequantization
get_k_full = '''ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    
    // TurboQuant: Dequantize if enabled
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
    
    auto * k = layers[ikv].k;
    const uint64_t kv_size = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];
    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    return ggml_view_4d(ctx, k,
        hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
        ggml_row_size(k->type, hparams.n_embd_head_k(il)),
        ggml_row_size(k->type, n_embd_k_gqa),
        ggml_row_size(k->type, n_embd_k_gqa*kv_size),
        ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}'''

get_k_pattern = r'ggml_tensor \* llama_kv_cache::get_k\(ggml_context \* ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo\) const \{[^}]+ggml_row_size\(k->type, n_embd_k_gqa\*kv_size\)\*sinfo\.s0\);\n\}'
content = re.sub(get_k_pattern, get_k_full, content)

with open('src/llama-kv-cache.cpp', 'w') as f:
    f.write(content)

print("Applied complete TurboQuant integration")
