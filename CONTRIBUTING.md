# Contributing to LoRA-EffSAM

Thanks for your interest in improving this project. Bug reports, fixes, documentation
improvements and well-scoped features are all welcome.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/FahimFBA/lora-effsam/issues/new/choose).
For bugs, include your OS, Python, PyTorch and CUDA versions, the exact command you ran,
and the full traceback. Report security problems privately, see [SECURITY.md](SECURITY.md).
Accessibility barriers can be reported as described in [ACCESSIBILITY.md](ACCESSIBILITY.md).

## Development setup

Linux or WSL2 with an NVIDIA GPU is needed for training and the GPU tests
(see the [README](README.md#install)).

```bash
conda create -n lora-effsam python=3.12 && conda activate lora-effsam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
```

Large artifacts (EfficientSAM weights, the Figshare dataset) are not tracked in git; use
the scripts in `scripts/` to fetch them.

## Making a change

1. Fork the repository and create a branch from `main`
   (`git checkout -b fix/short-description`).
2. Make focused changes. Keep unrelated refactors out of the same pull request.
3. Add or update tests in `tests/` for any behavior change.
4. Run the checks locally:

   ```bash
   ruff check .
   pytest
   ```

5. Update [CHANGELOG.md](CHANGELOG.md) for user-visible changes, and update the docs if
   behavior or CLI options change.
6. Open a pull request against `main` and fill in the template.

## Style

* Python 3.10+, linted with [ruff](https://docs.astral.sh/ruff/)
  (line length 110, see `pyproject.toml`).
* Follow the conventions of the surrounding code.
* Commit messages: short imperative subject line, optional body explaining the why
  (for example `fix: handle empty masks in dice loss`).

## Results and reproducibility

Changes that affect training, evaluation or reported numbers should state the seeds,
split and command used, and update the relevant records under `results/` where needed.
Do not commit datasets, model weights or other large binaries.

## License

By contributing you agree that your contributions are licensed under the
[Apache-2.0 License](LICENSE).
