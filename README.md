# BEACON: Language-Conditioned Navigation Affordance Prediction under Occlusion

Xinyu Gao, Gang Chen, Javier Alonso-Mora  
TU Delft

[![arXiv](https://img.shields.io/badge/arXiv-2603.09961-b31b1b.svg)](https://arxiv.org/abs/2603.09961)
[![IROS 2026](https://img.shields.io/badge/IROS-2026-1f6feb.svg)](https://iros2026.org/)

<p align="center">
  <img src="assets/images/overview.png" alt="BEACON overview" width="100%" />
</p>

## Release Status

- [x] BEACON model implementation
- [x] Evaluation code and metrics
- [x] Zero-shot baseline adapters
- [x] Data generation pipeline
- [ ] Processed BEACON dataset
- [ ] Pretrained checkpoints
- [ ] Verified reproduction environment

The remaining resources will be released in follow-up updates.

## Abstract

Language-conditioned local navigation requires a robot to infer a nearby traversable target location from its current observation and an open-vocabulary, relational instruction. Existing vision-language spatial grounding methods usually rely on vision-language models (VLMs) to reason in image space, producing 2D predictions tied to visible pixels. As a result, they struggle to infer target locations in occluded regions, typically caused by furniture or moving humans.

BEACON predicts an ego-centric Bird's-Eye View (BEV) affordance heatmap over a bounded local region, including occluded areas. Given an instruction and surround-view RGB-D observations from four directions around the robot, it combines ego-aligned vision-language reasoning with geometry-aware BEV features to infer traversable local targets. On an occlusion-aware Habitat benchmark, BEACON improves average geodesic accuracy by 22.74 percentage points over the strongest image-space baseline on the validation subset with occluded targets.

## Method Overview

BEACON uses two stages. First, an Ego-Aligned VLM learns to interpret spatial instructions in the robot frame using surround-view observations and depth-derived 3D position encoding. Second, the instruction-conditioned VLM representation is fused with a geometry-aware BEV encoder to predict a dense navigation affordance heatmap. The final target is the highest-scoring BEV location.

## Repository Structure

The repository is being organized around four public components:

```text
beacon/             BEACON model implementation
configs/            Training and evaluation configurations
evaluation/         Shared evaluation code and metrics
baselines/          Zero-shot baseline adapters
data_generation/    Social-MP3D-based data-generation pipeline
assets/             README and documentation assets
docs/               Installation and reproduction documentation
```

## Getting Started

### Installation

Environment recipes are being consolidated for three separate workflows: BEACON training, zero-shot baselines, and Habitat-based data generation. The final, verified reproduction environment will be released in a follow-up update.

### Evaluation and Zero-Shot Baselines

The release includes adapters for image-space zero-shot baselines and a shared evaluator for BEV target prediction. The evaluator reports geodesic accuracy, Euclidean accuracy, and structural invalid rate. Usage documentation and the processed evaluation data will be added in a follow-up update.

### BEACON Model

BEACON model code and experiment configurations are included in this release. Training and inference instructions will be finalized together with the processed dataset and pretrained checkpoints.

### Dataset

Processed BEACON data will be released in a follow-up update. Matterport3D assets must be obtained separately under their original terms of use.

### Data Generation

The data-generation pipeline is based on Habitat and Social-MP3D. It is intended for advanced users with access to the required simulator dependencies and licensed scene assets. Detailed documentation will be added as the pipeline is consolidated.

## Citation

```bibtex
@article{gao2026beacon,
  title={BEACON: Language-Conditioned Navigation Affordance Prediction under Occlusion},
  author={Gao, Xinyu and Chen, Gang and Alonso-Mora, Javier},
  journal={arXiv preprint arXiv:2603.09961},
  year={2026}
}
```

## Contact

For questions, please open a GitHub issue after the repository is published.

## Acknowledgements

This work builds on [XTuner](https://github.com/InternLM/xtuner), [Habitat](https://github.com/facebookresearch/habitat-lab), [Falcon](https://github.com/Zeying-Gong/Falcon), [RoboPoint](https://github.com/wentao-yuan/robopoint), and [RoboRefer](https://github.com/Alice-yli/RoboRefer). Third-party notices will be included with the corresponding release components.

## License

This project is released under the [MIT License](LICENSE).
