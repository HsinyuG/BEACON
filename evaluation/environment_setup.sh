#!/usr/bin/env bash
# BEACON evaluation / zero-shot baseline environment.
# Mirrors the recipe in evaluation/README.md (§Environment). Run from this
# directory (evaluation/), where pyproject.toml lives.
set -e

conda create -n beacon-baselines python=3.10 -y
conda activate beacon-baselines

pip install --upgrade pip

# Optional: a matching nvcc if you do not prefer the system one.
conda install -c nvidia cuda=12.1 -y

# Installs the vendored `robopoint` package (Apache-2.0) + the baseline adaptors.
pip install -e .

# Required by the unified evaluator (ExternalMetric).
pip install mmengine

# Required only by the GPT-4o baseline.
pip install openai

# NOTE: RoboRefer runs in its OWN environment from the upstream RoboRefer repo
# (see evaluation/README.md §RoboRefer-8B-SFT); it is not installed here.
