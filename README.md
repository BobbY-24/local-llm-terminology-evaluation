# Small local LLM research environment

This project runs a 4-bit Qwen2.5 3B **base** model locally with MLX-LM. The normal commands below use the local directory, raw completion prompts, greedy decoding, and no external API.

## Activate the environment

```bash
cd ~/llm-research
source .venv/bin/activate
```

## One-shot base-model generation

Use `--ignore-chat-template` for controlled raw completion. Keep prompts and generations short on an 8 GB Mac.

The downloaded base repository's tokenizer metadata does contain a chat template that inserts a helpful-assistant system message. That template is packaging metadata, not evidence that the weights are instruction-tuned. The explicit `--ignore-chat-template` flag prevents that scaffolding from contaminating the base condition. The Python scripts pass raw prompt text directly and never apply the template.

```bash
mlx_lm.generate \
  --model ./models/Qwen2.5-3B-4bit \
  --ignore-chat-template \
  --prompt $'Gaming glossary\n\nTerm: stat stick\nDefinition:' \
  --max-tokens 24 \
  --temp 0 \
  --seed 0 \
  --max-kv-size 512
```

Another raw completion:

```bash
mlx_lm.generate \
  --model ./models/Qwen2.5-3B-4bit \
  --ignore-chat-template \
  --prompt 'In online gaming, the term "smurf" refers to' \
  --max-tokens 24 \
  --temp 0 \
  --seed 0 \
  --max-kv-size 512
```

Because `--model` is a local path, these commands can be rerun with Wi-Fi disabled. The Python scripts additionally set Hugging Face and Transformers offline flags.

## Smoke test

```bash
python scripts/smoke_test.py
```

To supply another completion prompt:

```bash
python scripts/smoke_test.py --prompt 'In online gaming, the term "smurf" refers to'
```

## Interactive use

MLX-LM's `mlx_lm.chat` command always constructs chat messages. It is **not appropriate for controlled raw completion with this base checkpoint**, so no base-chat command is recommended. Re-run the one-shot command or `smoke_test.py --prompt ...` for each controlled base prompt.

If the matched instruction checkpoint is downloaded later to `./models/Qwen2.5-3B-Instruct-4bit`, its appropriate interactive command is:

```bash
mlx_lm.chat \
  --model ./models/Qwen2.5-3B-Instruct-4bit \
  --temp 0 \
  --seed 0 \
  --max-tokens 64 \
  --max-kv-size 512
```

Do not add `--ignore-chat-template` for that instruct chat command. Do not use `mlx_lm.chat` for a controlled base-vs-instruct comparison unless the research condition explicitly calls for chat formatting.

## Pilot evaluation

The data file demonstrates definition completion, natural continuation, and MCQ formats. Run:

```bash
python scripts/simple_eval.py
```

Results are saved under `results/` as timestamped JSONL. The script opens result files in exclusive-create mode, so it never overwrites a previous run. Every row records the raw prompt, output, model path and revision, quantization, runtime versions, decoding settings, and generation metrics.

Inspect the newest file with:

```bash
ls -lt results/
```

The defaults are explicit and near-deterministic: temperature 0 (greedy), top-p 1, top-k 0, seed 0, at most 32 new tokens, and a 512-token KV-cache cap. Change them only by passing command-line arguments; changed values are recorded in the output.

## Recreate the environment

The full installed package set is pinned in `requirements.txt`:

```bash
cd ~/llm-research
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The installed model itself is pinned to repository commit `76ae31041917ee0ef78284988aca339694acb7e9`. If it ever needs to be restored, download exactly that one snapshot:

```bash
hf download mlx-community/Qwen2.5-3B-4bit \
  --revision 76ae31041917ee0ef78284988aca339694acb7e9 \
  --local-dir ./models/Qwen2.5-3B-4bit
```

That restore command requires internet access; ordinary inference and evaluation do not.

## Interpretation cautions

- Four-bit quantization reduces memory and disk use but can alter token probabilities and lexical behavior relative to BF16.
- A base model continues text; it is not expected to reliably obey instructions, return one sentence, or select only an MCQ letter.
- Prompt whitespace, punctuation, decoding parameters, tokenizer version, checkpoint revision, and chat-template use are experimental variables.
- The base repository includes chat-template tokenizer metadata; accidentally omitting `--ignore-chat-template` from CLI generation changes the experimental condition.
- Greedy decoding removes sampling variation but does not guarantee bit-identical output across different MLX, macOS, or Apple Silicon versions.
- The 32K architectural context is not a sensible target on an 8 GB machine. These scripts cap the KV cache at 512 tokens and use very short prompts.
- Close memory-heavy apps before longer runs if macOS is already under memory pressure or actively swapping.

## Full community-terminology experiment

The systematic experiment uses the unchanged workbook copy at `data/raw/refined_terminology_usage_example_rich.xlsx`. It preserves the original pilot and adds deterministic dataset construction, resumable local inference, automatic scoring, term-cluster bootstrap analysis, SVG figures, and a blank human semantic-scoring artifact.

Run infrastructure tests:

```bash
cd ~/llm-research
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Build and validate the processed tasks:

```bash
python scripts/build_eval_dataset.py --config configs/terminology_eval.yaml
```

Run the three experiments sequentially. Each completed item is appended and flushed, and rerunning a command resumes from existing experiment IDs:

```bash
python scripts/run_terminology_eval.py \
  --config configs/terminology_eval.yaml \
  --experiment masked_recovery

python scripts/run_terminology_eval.py \
  --config configs/terminology_eval.yaml \
  --experiment context_compatibility

python scripts/run_terminology_eval.py \
  --config configs/terminology_eval.yaml \
  --experiment definition_ablation
```

Score and analyze:

```bash
python scripts/score_terminology_eval.py --config configs/terminology_eval.yaml
python scripts/analyze_terminology_eval.py --config configs/terminology_eval.yaml
```

The main report is `reports/terminology_eval_report.md`. Raw and scored JSONL files are under `results/terminology_eval/`; machine-readable tables are under `results/terminology_eval/summaries/tables/`; figures are under `figures/terminology_eval/`.

Definition outputs are not assigned automatic semantic scores. The discovered gold workbook is an unreviewed template with blank canonical meanings, so `results/terminology_eval/scored/terminology_eval_qwen25_3b_base_seed42_v1_definition_human_scoring.csv` leaves the 0/1/2 score and failure taxonomy fields blank for human review.
