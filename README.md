# Erebus v2 1.5B

A 1.5B parameter language model trained from scratch using the Qwen3 dense architecture.

**Status:** Training in progress on 8x A100-SXM4-80GB

## Model Details

| | |
|---|---|
| **Parameters** | ~1,720M |
| **Architecture** | Qwen3 Dense |
| **Context Length** | 2,048 tokens (4,096 max) |
| **Tokenizer** | Qwen3 (151K vocab) |
| **Precision** | bfloat16 |
| **Training Data** | 10B tokens |
| **HuggingFace** | [soyrsoyr/erebus-v2-1.5b-base](https://huggingface.co/soyrsoyr/erebus-v2-1.5b-base) |

## Architecture

Qwen3-style transformer with QK LayerNorm for stable training:

| | |
|---|---|
| Hidden Size | 2,048 |
| Intermediate Size | 6,144 |
| Layers | 28 |
| Attention Heads | 16 (8 KV heads, GQA 2:1) |
| Head Dim | 128 |
| Activation | SwiGLU |
| Normalization | RMSNorm (eps=1e-6) |
| Position Encoding | RoPE (theta=1M) |
| Attention Bias | None |
| Tied Embeddings | Yes |

## Tokenizer

This model uses the Qwen3 tokenizer (`Qwen/Qwen3-1.7B`, 151K vocab), matching the Qwen3 architecture. The larger vocabulary improves token efficiency over the Llama-2 32K tokenizer used in earlier runs.

## Training Data

Pre-tokenized mix totaling 10B tokens with quality filtering:

- **FineWeb-Edu** (score >= 4) — high-quality web text filtered for educational content
- **Cosmopedia v2** — synthetic textbook-style data
- **Python-Edu** (StarCoderData) — educational Python code, classifier-filtered to score >= 3

Quality pipeline (Gopher-inspired heuristics):
- Document length bounds (100 chars min, 100K max)
- Line-level and 10-gram repetition detection
- Special character ratio filtering
- Mean word length bounds
- Exact deduplication via SHA-256
- Unicode normalization and boilerplate stripping

Data is packed into fixed-length sequences with no padding, stored as memory-mapped uint16 binary files for efficient I/O.

## Training Configuration

| | |
|---|---|
| Optimizer | AdamW (fused) |
| Learning Rate | 3e-4 (cosine decay to 3e-5) |
| Batch Size | 512 sequences (8 batch x 8 grad_accum x 8 GPUs) |
| Tokens per Step | ~1M |
| Warmup | 2,000 steps (fixed) |
| Weight Decay | 0.1 |
| Betas | (0.9, 0.95) |
| Max Grad Norm | 10.0 |
| Gradient Checkpointing | Enabled |
| Mixed Precision | bf16 |

## Infrastructure

- **Hardware:** 8x NVIDIA A100-SXM4-80GB (single node)
- **Container:** `nvcr.io/nvidia/pytorch:24.12-py3` (PyTorch 2.6, CUDA 12.6)
- **Framework:** HuggingFace Transformers + Accelerate (DDP)
- **Platform:** OpenShift on bare-metal (IBM WDC cluster)

## Repository Structure

```
scripts/
  train.py          # Main training script
  eval.py           # Checkpoint evaluation and text generation
  prepare_data.py   # Data tokenization and packing pipeline
configs/
  accelerate_config.yaml
k8s/
  training-job.yaml # OpenShift job manifest
Dockerfile
requirements.txt
```

## Usage

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("soyrsoyr/erebus-v2-1.5b-base")
tokenizer = AutoTokenizer.from_pretrained("soyrsoyr/erebus-v2-1.5b-base")

inputs = tokenizer("The theory of relativity", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Training from Scratch

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 --standalone scripts/train.py \
    --data_dir /path/to/tokenized \
    --output_dir ./checkpoints \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 8
```

## Previous Version

[Erebus v1 (487M)](https://huggingface.co/soyrsoyr/erebus-487m-base) — Llama-style, trained on 529M tokens of FineWeb-Edu. This v2 1.5B model is a significant scale-up in both model size and data.

## License

Apache 2.0

## Author

[Sawyer Bowerman](https://huggingface.co/soyrsoyr)
