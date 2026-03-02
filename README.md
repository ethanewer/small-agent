# Project Layout

This repository is now split into two areas:

- `cli/` - active/new CLI codebase
- `sft/` - legacy SFT dataset and training-related code

## Working on the CLI

```bash
cd cli
uv sync
uv run terminus2-cli
```

## Working on Legacy SFT Code

```bash
cd sft
uv sync
uv run qwen-dataset-build --help
```
