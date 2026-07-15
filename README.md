# C-BELIEF-Stream

C-BELIEF-Stream is a research pipeline for temporally grounded evidence
attribution in renal deterioration assessment. It separates evidence available
at query time from future and retrospective evidence, and evaluates claim
verification, evidence attribution, temporal leakage, calibration, decision
curves, and subgroup performance.

## Repository scope

This GitHub repository contains code, configuration templates, and tests only.
Clinical and derived research data are deliberately excluded. See
[DATA_ACCESS.md](DATA_ACCESS.md) for the planned PhysioNet distribution and the
expected local directory layout.

## Requirements

- Python 3.10 or newer
- Authorized access to the separately distributed dataset
- Optional: a DeepSeek-compatible API or a local Hugging Face model for LLM runs

## Installation

Install the core package:

```bash
python -m pip install -e .
```

Install optional analysis, local-model, and development dependencies as needed:

```bash
python -m pip install -e ".[analysis]"
python -m pip install -e ".[local-llm]"
python -m pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` and provide credentials through environment variables. Never
put API keys in source files or committed configuration.

The default paths are defined in `configs/paths.yaml`. After obtaining authorized
data, place it under the ignored `data/` directory or update the configuration to
point to another protected location.

## Core pipeline

```bash
cbelief validate-data --config configs/paths.yaml
cbelief standardize-events --config configs/paths.yaml
cbelief build-streams --config configs/paths.yaml
cbelief build-samples --config configs/paths.yaml
cbelief split-data --config configs/paths.yaml --experiment configs/experiment.yaml
cbelief run-method --method cbelief --split val --config configs/paths.yaml
cbelief run-baseline --method rule_based --split val --config configs/paths.yaml
cbelief evaluate --split val --config configs/paths.yaml
```

## Agentic and LLM experiments

Agentic experiment settings live in `configs/agentic_cbelief.yaml`. API
credentials are read from environment variables. Local model paths may be set in
the configuration or through `CBELIEF_HF_MODEL`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Responsible use

This software is intended for research and reproducibility. It is not a medical
device and must not be used as a substitute for clinical judgment.

## Citation and license

Citation metadata, the PhysioNet dataset citation, and the software license must
be finalized before public release.
