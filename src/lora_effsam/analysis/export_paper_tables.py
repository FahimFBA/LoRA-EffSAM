# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
One row per training run with its full configuration and every test metric, for the
paper's experiment table.

    lora-effsam export-tables            # -> results/tables/all_runs.{csv,md}
Writes all_runs.csv and all_runs.md.
"""
import argparse
import csv
import json
import os
import statistics as st
from lora_effsam.paths import results_dir, runs_dir

RUNS = str(runs_dir())


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


def mean_dice(report):
    return st.mean(p["dice"] for p in report["per_image"]) if report else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(results_dir() / "tables"))
    out = ap.parse_args(argv).out
    os.makedirs(out, exist_ok=True)

    rows = []
    for name in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, name)
        cfg = load(os.path.join(d, "config.json"))
        if not os.path.isdir(d) or cfg is None:
            continue
        a, ma = cfg.get("args", {}), cfg.get("model_args", {})
        test = load(os.path.join(d, "test.json")) or {}
        with open(os.path.join(d, "metrics.csv")) as f:
            m = list(csv.DictReader(f))
        hours = sum(float(r["epoch_minutes"]) for r in m) / 60
        mc = load(os.path.join(d, "uncertainty_report_ids.json")) or load(os.path.join(d, "uncertainty_report.json"))
        tta = load(os.path.join(d, "uncertainty_report_tta.json"))
        rows.append(dict(
            run=name, seed=a.get("seed", 42),
            encoder=ma.get("variant", "vitt"), decoder=ma.get("decoder", "simple"),
            vit_skips=int(bool(ma.get("vit_skips"))), deep_supervision=int(bool(ma.get("deep_supervision"))),
            lora_mlp=int(bool(ma.get("lora_mlp"))), lora_rank=ma.get("lora_r"),
            image_size=ma.get("image_size"), encoder_size=ma.get("encoder_size"),
            lr=a.get("lr"), batch=f"{a.get('batch_size')}x{a.get('grad_accum_steps')}",
            intensity_aug=int(bool(a.get("intensity_aug"))),
            size_T=cfg.get("split", {}).get("small_lesion_threshold"), size_tau=a.get("tau", 20.0),
            epochs_planned=a.get("epochs", cfg.get("epochs_planned")), epochs_trained=len(m),
            best_epoch=test.get("best_epoch"), best_val_dice=test.get("best_val_dice"),
            test_dice=test.get("test_dice"), test_dice_mc=mean_dice(mc), test_dice_mc_tta=mean_dice(tta),
            trainable_params=cfg.get("trainable_parameters", ""), train_hours=round(hours, 2),
            config_source="reconstructed" if cfg.get("args_reconstructed_from_run_script") else "logged",
        ))

    keys = list(rows[0])
    with open(os.path.join(out, "all_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and v < 1 else ("" if v is None else str(v))
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    open(os.path.join(out, "all_runs.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
