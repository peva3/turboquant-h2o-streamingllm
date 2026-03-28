# GPU Benchmark Script for TurboQuant

## Prerequisites

1. CUDA build must be complete:
```bash
cd /app/turbo/llama.cpp/build
cmake .. -DGGML_CUDA=ON -DGGML_CUDA_ARCHS="89"
make -j2  # This takes 15-20 minutes
```

2. GPU must be detected:
```bash
./bin/llama-cli --list-devices
# Should show: NVIDIA GeForce RTX 4060 Ti
```

## Quick Benchmark

```bash
#!/bin/bash
MODEL="/app/turbo/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"

echo "=== GPU TurboQuant Benchmark ==="
echo ""
echo "1. Baseline (no TurboQuant):"
./bin/llama-cli -m "$MODEL" -p "Test:" -n 100 -ngl 99 2>&1 | grep "t/s"

echo ""
echo "2. With TurboQuant:"
./bin/llama-cli -m "$MODEL" -p "Test:" -n 100 --turboquant -ngl 99 2>&1 | grep "t/s"
```

## Full Benchmark Suite

### Test 1: Speed Comparison
```bash
MODEL="/app/turbo/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"

# Short context (4K)
./bin/llama-bench -m "$MODEL" -p 512 -n 128 -ngl 99 2>&1 | grep -E "t/s|model"
./bin/llama-bench -m "$MODEL" -p 512 -n 128 --turboquant -ngl 99 2>&1 | grep -E "t/s|model"

# Long context (16K)
./bin/llama-cli -m "$MODEL" -c 16384 -p "Test:" -n 200 -ngl 99 2>&1 | grep "t/s"
./bin/llama-cli -m "$MODEL" -c 16384 -p "Test:" -n 200 --turboquant -ngl 99 2>&1 | grep "t/s"
```

### Test 2: Memory Comparison
```bash
# Monitor VRAM usage
watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv'

# In another terminal:
./bin/llama-cli -m "$MODEL" -n 1000 -ngl 99  # Baseline
./bin/llama-cli -m "$MODEL" -n 1000 --turboquant -ngl 99  # TurboQuant
```

### Test 3: Quality Check
```bash
# Generate text and compare quality
./bin/llama-cli -m "$MODEL" -p "Write a story about AI:" -n 200 -ngl 99 > baseline.txt
./bin/llama-cli -m "$MODEL" -p "Write a story about AI:" -n 200 --turboquant -ngl 99 > turboquant.txt

# Compare outputs
diff baseline.txt turboquant.txt
```

## Expected Results

### Speed (RTX 4060 Ti)
| Context | Baseline | TurboQuant | Speedup |
|---------|----------|------------|---------|
| 4K | ~15 t/s | ~45 t/s | 3.0x |
| 8K | ~8 t/s | ~30 t/s | 3.75x |
| 16K | ~4 t/s | ~20 t/s | 5.0x |

### Memory (VRAM Usage)
| Context | Baseline | TurboQuant | Saved |
|---------|----------|------------|-------|
| 4K | 6 GB | 2 GB | 4 GB |
| 8K | 10 GB | 3 GB | 7 GB |
| 16K | 18 GB* | 5 GB | 13 GB |
| 32K | OOM | 9 GB | N/A |

*Requires CPU offloading without TurboQuant

## Troubleshooting

### CUDA build not complete
```bash
cd /app/turbo/llama.cpp/build
make ggml-cuda -j2
```

### GPU not detected
```bash
# Check nvidia-smi
nvidia-smi

# Check CUDA path
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Out of memory
```bash
# Reduce context size
./bin/llama-cli -m model.gguf -c 2048 -ngl 99

# Or increase eviction
./bin/llama-cli -m model.gguf --turboquant --evict-ratio 0.2 -ngl 99
```
