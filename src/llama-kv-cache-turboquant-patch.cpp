// TurboQuant integration patch for llama-kv-cache.cpp
// This file contains the modified functions with actual TurboQuant usage

// Modified get_k - dequantizes from quantized buffer when TurboQuant enabled
ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    
    // Check if TurboQuant is enabled and we have quantized data
    if (layers[ikv].use_turboquant && layers[ikv].key_quantized_data && layers[ikv].key_turboquant) {
        const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
        const size_t buffer_offset = sinfo.s0;
        const size_t bytes_per_vector = layers[ikv].bytes_per_vector;
        
        // Create output tensor
        ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns);
        
        // Dequantize from quantized buffer
        const uint8_t* quantized_base = layers[ikv].key_quantized_data;
        float* output_ptr = (float*)result->data;
        const int64_t vector_size = hparams.n_embd_head_k(il) * hparams.n_head_kv(il);
        
        for (int64_t i = 0; i < (int64_t)n_kv * ns; i++) {
            const uint8_t* quantized_data = quantized_base + (buffer_offset + i) * bytes_per_vector;
            layers[ikv].key_turboquant->dequantize_to_buffer(
                quantized_data,
                output_ptr + i * vector_size,
                vector_size
            );
        }
        
        return result;
    }
    
    // Fallback to standard implementation
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
}

// Modified get_v - dequantizes from quantized buffer when TurboQuant enabled  
ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    
    // Check if TurboQuant is enabled and we have quantized data
    if (layers[ikv].use_turboquant && layers[ikv].value_quantized_data && layers[ikv].value_turboquant) {
        const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
        const size_t buffer_offset = sinfo.s0;
        const size_t bytes_per_vector = layers[ikv].bytes_per_vector;
        
        // Create output tensor
        ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns);
        
        // Dequantize from quantized buffer
        const uint8_t* quantized_base = layers[ikv].value_quantized_data;
        float* output_ptr = (float*)result->data;
        const int64_t vector_size = hparams.n_embd_head_v(il) * hparams.n_head_kv(il);
        
        for (int64_t i = 0; i < (int64_t)n_kv * ns; i++) {
            const uint8_t* quantized_data = quantized_base + (buffer_offset + i) * bytes_per_vector;
            layers[ikv].value_turboquant->dequantize_to_buffer(
                quantized_data,
                output_ptr + i * vector_size,
                vector_size
            );
        }
        
        return result;
    }
    
    // Fallback to standard implementation
    auto * v = layers[ikv].v;
    const uint64_t kv_size = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    
    if (!v_trans) {
        return ggml_view_4d(ctx, v,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(v->type, hparams.n_embd_head_v(il)),
            ggml_row_size(v->type, n_embd_v_gqa),
            ggml_row_size(v->type, n_embd_v_gqa*kv_size),
            ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }
    
    return ggml_view_4d(ctx, v,
        n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
        ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),
        ggml_row_size(v->type, kv_size),
        ggml_row_size(v->type, kv_size*n_embd_v_gqa),
        ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}
