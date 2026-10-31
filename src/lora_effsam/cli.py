# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""Command-line entry point: `lora-effsam <subcommand> ...`.

Each subcommand is a thin wrapper: the remaining arguments are passed unchanged to the
``main(argv)`` of the corresponding module, so `lora-effsam <subcommand> --help` shows that
module's own options and every step is also usable programmatically (e.g. from a
notebook) by importing the module directly.
"""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys

# subcommand -> (module, one-line description), in pipeline order
COMMANDS = {
    "verify-env": (None, "Check Python, PyTorch, CUDA and the data/EfficientSAM/runs locations"),
    "prepare-data": ("lora_effsam.data.mat_to_dataset", "Convert the raw Figshare .mat files to PNG + metadata.csv"),
    "train": ("lora_effsam.training.train", "Train one model (writes config.json, metrics.csv, test.json)"),
    "evaluate": ("lora_effsam.evaluation.evaluate_uncertainty", "MC-Dropout test evaluation (+ --tta) with per-image report"),
    "visualize": ("lora_effsam.evaluation.visualize", "Input / ground truth / prediction / uncertainty figure"),
    "summarize": ("lora_effsam.analysis.summarize_runs", "Summary tables over all runs (mean ± SD over seeds)"),
    "export-tables": ("lora_effsam.analysis.export_paper_tables", "One row per run with full config + metrics"),
    "paper-stats": ("lora_effsam.analysis.paper_stats", "Paired statistical tests + learning-curve/subgroup figures"),
    "uncertainty-stats": ("lora_effsam.analysis.uncertainty_stats", "MC-Dropout uncertainty analysis"),
    "qualitative-figure": ("lora_effsam.analysis.qualitative_figure", "Rule-selected qualitative cases"),
    "explainability": ("lora_effsam.analysis.explainability_figures", "Architecture, internal maps, uncertainty, Dice figures"),
    "knockout": ("lora_effsam.analysis.knockout_analysis", "Test-time component knock-outs"),
    "failure-check": ("lora_effsam.analysis.failure_level_check", "At which decoder level failures are decided"),
}


def cmd_verify_env() -> int:
    """Runs ``lora-effsam verify-env``: prints the environment and the configured locations.

    :return: 0 if PyTorch imports and CUDA is available, else 1.
    :rtype: int
    """
    from lora_effsam import paths

    report = {"python": platform.python_version(), "nvidia_smi": shutil.which("nvidia-smi") is not None}
    try:
        import torch

        report.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        report.update(torch=None, cuda=False)
    for name in ("data_dir", "runs_dir", "efficientsam_dir", "results_dir"):
        p = getattr(paths, name)()
        report[name] = {"path": str(p), "exists": p.exists()}
    if report["nvidia_smi"]:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        report["nvidia_smi_gpus"] = out.stdout.strip().splitlines()
    print(json.dumps(report, indent=2))
    return 0 if report.get("cuda") else 1


def _usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = ["usage: lora-effsam <subcommand> [options]   (lora-effsam <subcommand> --help for its options)", "",
             "subcommands:"]
    lines += [f"  {c:<{width}}  {desc}" for c, (_, desc) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Dispatches ``lora-effsam <subcommand> [args...]`` to the subcommand's module.

    :param argv: Command-line arguments without the program name (default: ``sys.argv[1:]``).
    :return: Process exit code.
    :rtype: int
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"lora_effsam: unknown subcommand {cmd!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    if cmd == "verify-env":
        return cmd_verify_env()
    module = importlib.import_module(COMMANDS[cmd][0])
    result = module.main(rest)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
