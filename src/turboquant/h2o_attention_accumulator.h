#pragma once

#include <cstdint>

struct ggml_tensor;
struct ggml_context;

namespace turboquant {

// H2O Attention Score Accumulator
// Accumulates attention scores across heads for each KV position.
// This is used for the Heavy-Hitter Oracle eviction policy.
class H2OAttentionAccumulator {
public:
    // Constructor
    // scores: pointer to an array of n_kv floats to accumulate into
    // n_kv: number of KV positions (cache size)
    // n_head: number of attention heads
    H2OAttentionAccumulator(float* scores, int n_kv, int n_head);
    
    ~H2OAttentionAccumulator();
    
    // Reset all accumulated scores to zero
    void reset();
    
    // Get the accumulated scores array
    const float* get_scores() const { return accumulated_scores_; }
    float* get_scores_mut() { return accumulated_scores_; }
    
    // Get the number of KV positions
    int get_n_kv() const { return n_kv_; }
    
    // Get the number of heads
    int get_n_head() const { return n_head_; }

private:
    int n_kv_;
    int n_head_;
    float* accumulated_scores_;  // [n_kv] - cumulative attention scores per position (not owned)
};

} // namespace turboquant

#ifdef __cplusplus
extern "C" {
#endif

// Create a custom op that accumulates attention scores from a softmax output.
// The input tensor should be the attention weights after softmax.
// kq: [n_head, n_kv, n_tokens] - attention weights after softmax
// accumulator: H2OAttentionAccumulator instance
// Returns: the same tensor (identity operation, but accumulates scores as side effect)
ggml_tensor* ggml_h2o_accumulate_scores(
    ggml_context* ctx,
    ggml_tensor* kq,
    turboquant::H2OAttentionAccumulator* accumulator);

#ifdef __cplusplus
}
#endif
