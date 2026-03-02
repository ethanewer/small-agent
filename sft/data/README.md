# Dataset Build Guide

This directory contains all dataset pipeline code and generated dataset outputs.

## Dataset modes

- `demo`: 100 rows per source for quick validation.
- `full`: include all NVIDIA terminal splits + all Nanbeige + all SWE.
- `balanced`: include all Nanbeige + all SWE, then select NVIDIA to target ~50% NVIDIA bytes.

## Sources used

- `nvidia/Nemotron-Terminal-Corpus`
  - `skill_based_easy/train`
  - `skill_based_medium/train`
  - `skill_based_mixed/train`
- `Nanbeige/ToolMind-Web-QA` (JSONL files only)
- `SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K`

## Rebuild from scratch

From the `sft/` directory:

```bash
uv sync
```

### Demo dataset

```bash
uv run qwen-dataset-build --mode demo \
  --rows-per-source 100 \
  --demo-output-dir "/wbl-fast/usrs/ethan/small-agent/sft/data/demo-dataset"
```

### Full dataset

```bash
uv run qwen-dataset-build --mode full \
  --full-output-dir "/wbl-fast/usrs/ethan/small-agent/sft/data/full-dataset"
```

### Additional Nemotron dataset (one-off, capped)

Build up to 10k rows per split from:

- `nvidia/Nemotron-Agentic-v1` (`interactive_agent`, `tool_calling`)
- `nvidia/Nemotron-Instruction-Following-Chat-v1` (`chat_if`, `structured_outputs`)

```bash
uv run qwen-dataset-build --mode additional \
  --additional-rows-per-split 10000 \
  --additional-output-dir "/wbl-fast/usrs/ethan/small-agent/sft/data/additional-data"
```

By default this keeps memory lower by writing only `dataset_shuffled.jsonl` + `metadata.json`.
Add `--materialize-hf-dataset true` only if you also need `dataset/`.

### Extended full dataset (one-off merge)

Merge existing full JSONL with additional JSONL into a new output directory:

```bash
uv run qwen-dataset-build --mode extended_full \
  --base-full-input-jsonl "/wbl-fast/usrs/ethan/small-agent/sft/data/full-dataset/dataset_shuffled.jsonl" \
  --additional-input-jsonl "/wbl-fast/usrs/ethan/small-agent/sft/data/additional-data/dataset_shuffled.jsonl" \
  --extended-output-dir "/wbl-fast/usrs/ethan/small-agent/sft/data/full-dataset-extended"
```

This mode also defaults to JSONL-only output to avoid OOM risk on large merges.

### Balanced dataset (~50% NVIDIA bytes)

```bash
uv run qwen-dataset-build --mode balanced \
  --balanced-output-dir "/wbl-fast/usrs/ethan/small-agent/sft/data/balanced-dataset"
```

## Output artifacts

Each dataset directory contains:

- `dataset/` - Hugging Face dataset saved to disk.
- `metadata.json` - source counts/bytes, policy details, and reproducibility fields.
- `dataset_shuffled.jsonl` (full/balanced) - deterministic shuffled pre-load JSONL.

Additional/extended one-off modes write:

- `dataset_shuffled.jsonl` for `additional-data` and `full-dataset-extended`
- `metadata.json`
- optional `dataset/` when `--materialize-hf-dataset true` is set

Demo mode also writes:

- `examples.jsonl` - chat-template-rendered examples using Qwen tokenizer/template.
- `source_samples.jsonl` - cleaned sample rows.

## Metadata fields for reproducibility

`metadata.json` includes:

- `seed`
- `source_specs`
- `rows_by_source`
- `bytes_by_source`
- `rows_by_source_before_balance`
- `bytes_by_source_before_balance`
- mode-specific policy fields (for example `nvidia_actual_byte_share`)

## Optional remote preprocessing helpers

Scripts are in `data/scripts/`:

- `data/scripts/check_nodes.sh`
- `data/scripts/preprocess_remote.sh`

Usage:

```bash
./data/scripts/check_nodes.sh /path/to/krafton-dld-public.pem
./data/scripts/preprocess_remote.sh /path/to/krafton-dld-public.pem
```

Remote helper notes:

- It distributes per-source shards across available nodes.
- It uses current source policy (all NVIDIA terminal splits, Nanbeige JSONL handling, SWE).
- It syncs partial JSONL outputs to `data/remote-partials/`.

## Caveats

- Set `HF_TOKEN` for higher Hugging Face rate limits.
- No token-length filtering is done here; filter at training time.
- Think-tag cleanup is applied before writing rows.
- `extended_full` mode expects `full-dataset/dataset_shuffled.jsonl` and `additional-data/dataset_shuffled.jsonl` to exist.
