# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

To publish a release: add a `## [X.Y.Z] - YYYY-MM-DD` section below, bump `version` in
`pyproject.toml`, then push a tag `vX.Y.Z`. The release workflow publishes that section as the
GitHub release notes.

## [Unreleased]

## [0.2.0] - 2026-10-31

### Added
- First public release of LoRA-EffSAM: parameter-efficient brain-tumor MRI segmentation with a
  LoRA-adapted EfficientSAM encoder, hybrid CNN-ViT decoder, size-adaptive boundary-aware loss,
  MC-Dropout uncertainty and explainability analysis.
- `lora-effsam` CLI with stages for data preparation, training, evaluation, statistics,
  uncertainty analysis and explainability.
- Scripts to fetch EfficientSAM and the Figshare dataset, run the full pipeline and reproduce
  every table, statistic and figure in `results/`.
- Experiment records, tables and figures for all reported runs in `results/`.
