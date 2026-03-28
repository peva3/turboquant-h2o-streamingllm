with open('src/llama-kv-cache.cpp', 'r') as f:
    content = f.read()

# Find where layers are initialized and add TurboQuant initialization
old_layer_init = '''layers.push_back({ il, k, v, k_stream, v_stream, });'''

new_layer_init = '''// Initialize TurboQuant state
kv_layer new_layer(il, k, v, k_stream, v_stream);
new_layer.use_turboquant = use_turboquant;
new_layer.key_turboquant = nullptr;
new_layer.value_turboquant = nullptr;
new_layer.key_quantized_data = nullptr;
new_layer.value_quantized_data = nullptr;
new_layer.key_quantized_size = 0;
new_layer.value_quantized_size = 0;
new_layer.bytes_per_vector = 0;

// Allocate quantized buffers if TurboQuant is enabled
if (use_turboquant && k) {
    int embedding_dim = k->ne[0];
    int kv_size = k->ne[1];
    int bit_width = 4;
    int bits_per_vector = embedding_dim * bit_width + 1;
    size_t bytes_per_vector = (bits_per_vector + 7) / 8;
    
    new_layer.bytes_per_vector = bytes_per_vector;
    new_layer.key_quantized_size = kv_size * bytes_per_vector;
    new_layer.value_quantized_size = kv_size * bytes_per_vector;
    new_layer.key_quantized_data = new uint8_t[new_layer.key_quantized_size]();
    new_layer.value_quantized_data = new uint8_t[new_layer.value_quantized_size]();
    
    // Initialize TurboQuant objects
    new_layer.key_turboquant = std::make_unique<turboquant::TurboQuantKVCache>(embedding_dim, bit_width, 42);
    new_layer.value_turboquant = std::make_unique<turboquant::TurboQuantKVCache>(embedding_dim, bit_width, 42);
    
    new_layer.key_turboquant->init();
    new_layer.value_turboquant->init();
}

layers.push_back(new_layer);'''

content = content.replace(old_layer_init, new_layer_init)

# Also need to add destructor to free buffers
destructor_addition = '''
// Free TurboQuant buffers
for (auto & layer : layers) {
    if (layer.key_quantized_data) {
        delete[] layer.key_quantized_data;
        layer.key_quantized_data = nullptr;
    }
    if (layer.value_quantized_data) {
        delete[] layer.value_quantized_data;
        layer.value_quantized_data = nullptr;
    }
}
'''

# Find destructor and add cleanup
if '~llama_kv_cache' in content:
    # Destructor exists, add cleanup before closing brace
    destructor_pattern = 'llama_kv_cache::~llama_kv_cache() {'
    if destructor_pattern in content:
        content = content.replace(destructor_pattern, destructor_addition + 'llama_kv_cache::~llama_kv_cache() {')

with open('src/llama-kv-cache.cpp', 'w') as f:
    f.write(content)

print("Added TurboQuant initialization and cleanup")
