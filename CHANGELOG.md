# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

To publish a release: add a `## [X.Y.Z] - YYYY-MM-DD` section below, bump `version` in
`pyproject.toml`, then push a tag `vX.Y.Z`. The release workflow publishes that section as the
GitHub release notes.

## [Unreleased]

### Added
- Project website in `docs/`: abstract, method, interactive results chart, interpretation,
  figures, run instructions and a copyable citation.
- `.github/workflows/pages.yml` deploys `docs/` to GitHub Pages on every change to the site.

### Changed
- `CITATION.cff` and the README citation now point to the paper, with its full seven-author
  list and title. The software authors in `CITATION.cff` are unchanged.

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
