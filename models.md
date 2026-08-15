# Local model inventory and reproducibility metadata

Inventory date: 2026-08-15

## Existing models found before setup

| Model | Stage | Parameters | Quantization | Disk size | Location | Runtime | 8 GB M3 suitability | Evaluation value |
|---|---|---:|---|---:|---|---|---|---|
| `qwen3:1.7b` | Post-trained/chat | Ollama reports model type 2.0B; marketed as 1.7B | GGUF `Q4_K_M` | 1,359,279,776-byte model blob; about 1.27 GiB | `~/.ollama/models/` | Ollama 0.32.5 | Very comfortable | Useful post-trained lightweight reference, but not a matched 3B-4B base model |
| `gemma3:1b` | Instruction/chat | Ollama reports 999.89M | GGUF `Q4_K_M` | 815,310,432-byte model blob; about 0.76 GiB | `~/.ollama/models/` | Ollama 0.32.5 | Very comfortable | Useful small chat baseline, but not a Qwen base/instruct comparison |

No Hugging Face cache, LM Studio model directory, shallow-home GGUF/safetensors file, or obvious llama.cpp model directory was found. Existing files were not changed.

## Runtime/tool inventory before setup

- `python3`: installed at `/opt/homebrew/bin/python3`, version 3.14.6
- `pip`: installed for the Python.org 3.13 framework; `pip3` also installed via Homebrew
- `mlx`: framework 0.32.0 installed globally via Homebrew/Python, but no `mlx` executable was on `PATH`
- `mlx-lm` / `mlx_lm.generate` / `mlx_lm.chat`: not installed before this setup
- `ollama`: installed, version 0.32.5; its background service was not running during inventory
- `llama.cpp` / `llama-cli`: not found on `PATH`
- `huggingface-cli` / `hf`: not found on `PATH` before setup; `hf` is now available only inside this project's virtual environment as a dependency

## Installed model

- Model family: Qwen2.5
- Exact checkpoint: Qwen2.5-3B base, MLX 4-bit conversion
- Repository/model ID: `mlx-community/Qwen2.5-3B-4bit`
- Base vs instruct: base/pretrained
- Parameter count: approximately 3.09B
- Quantization: 4-bit weights, group size 64; some non-quantized tensors remain in BF16
- Runtime: MLX-LM 0.31.3 / MLX 0.32.0
- Python: 3.13.3, isolated virtual environment
- macOS / Apple Silicon: macOS 26.5.2 (build 25F84), arm64, Apple M3, 8 GiB unified memory
- Model architecture context limit: 32,768 tokens
- Evaluation KV-cache cap: 512 tokens
- Default temperature: 0.0 (greedy)
- Default top-p: 1.0
- Default top-k: 0
- Default seed: 0
- Default maximum new tokens: 32 for pilot evaluation, 24 for smoke test
- Chat template applied: no
- Local model path: `~/llm-research/models/Qwen2.5-3B-4bit`
- Repository revision: `76ae31041917ee0ef78284988aca339694acb7e9`
- Date downloaded: 2026-08-15
- Weight file: 1,736,293,090 bytes
- Local repository size: approximately 1.75 GB decimal / 1.63 GiB

## Selection decision

`Qwen/Qwen3-4B-Base` exists, and established official/MLX Community 4-bit conversions exist for the post-trained Qwen3-4B model. An established official/MLX Community 4-bit conversion of the *base* Qwen3-4B checkpoint could not be verified. A few third-party base conversions exist, but using one would introduce an avoidable conversion/provenance difference.

`mlx-community/Qwen2.5-3B-4bit` was therefore selected as the closest established 3B-4B Qwen base checkpoint. It has a directly matched later comparison checkpoint: `mlx-community/Qwen2.5-3B-Instruct-4bit`. The instruct checkpoint was identified but not downloaded, avoiding another roughly 1.75 GB of storage until the base pilot is accepted.

## Installed Python packages

Exact environment pins are in `requirements.txt`. Key versions are MLX-LM 0.31.3, MLX 0.32.0, Transformers 5.15.0, Hugging Face Hub 1.27.0, and NumPy 2.5.2.

## Smoke-test observation

- Prompt: `Gaming glossary\n\nTerm: stat stick\nDefinition:`
- Generated 24 tokens successfully using the local model path with offline flags set.
- Output began: `A stat stick is a small, handheld device that is used to track the performance of players during a game.` This is fluent but semantically wrong for the intended gaming sense, which is useful evidence rather than a setup failure.
- Generation rate: approximately 44.15 tokens/second for this short run.
- MLX-reported peak memory: 1.761 GB; process peak memory footprint reported by macOS: approximately 1.98 GB.
- System swap before: 7,523.94 MB used. System swap after: 9,115.69 MB used, an increase of about 1,591.75 MB during model load/generation.
- Conclusion: the model itself loads and runs comfortably, but the already-busy 8 GB system performed significant system-wide swapping to make room. Close Chrome, IDEs, and other memory-heavy apps before longer experimental runs.
