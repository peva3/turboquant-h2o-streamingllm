#include "h2o_attention_accumulator.h"
extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
}
#include <cstring>
#include <algorithm>

namespace turboquant {

H2OAttentionAccumulator::H2OAttentionAccumulator(float* scores, int n_kv, int n_head)
     : n_kv_(n_kv), n_head_(n_head), accumulated_scores_(scores) {
     // Note: We don't own the scores array, just use it
     // The array should already be zeroed by the caller
 }

H2OAttentionAccumulator::~H2OAttentionAccumulator() {
    // Note: We don't own the scores array, just use it
    // The array is managed by the KV cache layer
}

void H2OAttentionAccumulator::reset() {
    std::memset(accumulated_scores_, 0, n_kv_ * sizeof(float));
}

// Forward function for the custom op
static void h2o_accumulate_forward(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void* userdata) {
    H2OAttentionAccumulator* acc = static_cast<H2OAttentionAccumulator*>(userdata);
    if (!acc) {
        return;
    }
    
    // Input tensor shape: [n_head, n_kv, n_tokens] after softmax
    // We need to sum across n_head and n_tokens for each KV position
    const int64_t n_head = a->ne[2];  // usually this is dimension 2
    const int64_t n_kv = a->ne[1];    // KV positions
    const int64_t n_tokens = a->ne[0]; // query tokens
    
    // Handle different tensor layouts
    // Standard attention output: [n_tokens, n_kv, n_head] or [n_head, n_kv, n_tokens]
    // After ggml_soft_max, the tensor is typically [n_kv, n_tokens, n_head] or similar
    
    const float* src_data = (const float*)a->data;
    const int64_t nb0 = a->nb[0];
    const int64_t nb1 = a->nb[1];
    const int64_t nb2 = a->nb[2];
    
    float* scores = acc->get_scores_mut();
    
    // Partition work across KV positions
    const int64_t start = (ith * n_kv) / nth;
    const int64_t end = ((ith + 1) * n_kv) / nth;
    
    // Sum attention weights for each KV position
    // For each KV position, we sum across all query tokens and all heads
    for (int64_t kv_pos = start; kv_pos < end; ++kv_pos) {
        float sum = 0.0f;
        for (int64_t head = 0; head < n_head; ++head) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                // Access the element at (token, kv_pos, head)
                // The exact indexing depends on the tensor layout
                const float* ptr = (const float*)((const char*)src_data + 
                    token * nb0 + kv_pos * nb1 + head * nb2);
                sum += *ptr;
            }
        }
        // Atomically add to accumulated scores
        // Note: This is thread-safe because each thread handles different KV positions
        scores[kv_pos] += sum;
    }
    
    // Copy input to output (identity operation)
    if (dst && dst->data && a->data) {
        // Only copy if we're the first thread
        if (ith == 0) {
            size_t total_size = ggml_nbytes(a);
            std::memcpy(dst->data, a->data, total_size);
        }
    }
}

} // namespace turboquant

ggml_tensor* ggml_h2o_accumulate_scores(
    ggml_context* ctx,
    ggml_tensor* kq,
    turboquant::H2OAttentionAccumulator* accumulator) {
    
    // Create identity output tensor
    ggml_tensor* result = ggml_map_custom1(
        ctx,
        kq,
        turboquant::h2o_accumulate_forward,
        GGML_N_TASKS_MAX,
        static_cast<void*>(accumulator)
    );
    
    result->op = GGML_OP_CUSTOM;
    
    return result;
}
