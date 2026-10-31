# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Collect every experiment under runs/ into one comparison table for the paper.

    lora-effsam summarize                # writes runs/summary.csv + runs/summary.md
    lora-effsam summarize --backfill     # first rebuild metrics.csv/test.json from
                                                    # <run>.train.log for runs trained before
                                                    # train.py wrote them

Per run it reads config.json, metrics.csv, test.json and uncertainty_report.json
(per-class and per-size-quartile test Dice come from the per-image report).
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

from lora_effsam.config import TUMOR_CLASSES
from lora_effsam.paths import runs_dir

EPOCH_RE = re.compile(
    r"^\[(\d+)/(\d+)\] lr=([\d.e+-]+) train_loss=([\d.]+) train_dice=([\d.]+) \| "
    r"val_loss=([\d.]+) val_dice=([\d.]+) \(([\d.]+) min\)")
TEST_RE = re.compile(r"^TEST \(best\.pt\): loss=([\d.]+) dice=([\d.]+) \(n=(\d+)\)")
FIELDS = ["epoch", "lr", "train_loss", "train_dice", "val_loss", "val_dice",
          "epoch_minutes", "is_best", "timestamp"]


def backfill(run_dir):
    """Rebuild metrics.csv / test.json / config.json from the text log (never overwrites)."""
    log = run_dir.rstrip("/") + ".train.log"
    if not os.path.isfile(log):
        return
    rows, test, best = [], None, -1.0
    for line in open(log):
        m = EPOCH_RE.match(line)
        if m:
            # The logged lr is the value AFTER scheduler.step(), i.e. next epoch's lr.
            e, total, lr, tl, td, vl, vd, mins = m.groups()
            vd = float(vd)
            rows.append(dict(epoch=int(e), lr=float(lr), train_loss=float(tl),
                             train_dice=float(td), val_loss=float(vl), val_dice=vd,
                             epoch_minutes=float(mins), is_best=int(vd > best), timestamp=""))
            best = max(best, vd)
        m = TEST_RE.match(line)
        if m:
            test = dict(test_loss=float(m[1]), test_dice=float(m[2]), n_test=int(m[3]))

    metrics = os.path.join(run_dir, "metrics.csv")
    if rows and not os.path.isfile(metrics):
        with open(metrics, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"  {metrics}: {len(rows)} epochs (from log; lr column is post-step)")

    test_path = os.path.join(run_dir, "test.json")
    if test and rows and not os.path.isfile(test_path):
        best_row = max(rows, key=lambda r: r["val_dice"])
        test.update(best_epoch=best_row["epoch"], best_val_dice=best_row["val_dice"],
                    last_epoch=rows[-1]["epoch"], reconstructed_from_log=True)
        json.dump(test, open(test_path, "w"), indent=2)
        print(f"  {test_path}")

    config_path = os.path.join(run_dir, "config.json")
    best_pt = os.path.join(run_dir, "best.pt")
    if not os.path.isfile(config_path) and os.path.isfile(best_pt):
        import torch
        ck = torch.load(best_pt, map_location="cpu", weights_only=False)
        json.dump({"model_args": ck["model_args"], "split": ck["split"],
                   "epochs_planned": int(EPOCH_RE.match(
                       next(l for l in open(log) if EPOCH_RE.match(l)))[2]),
                   "reconstructed_from_checkpoint": True,
                   "note": "lr / batch / patience not stored in old checkpoints; see run script"},
                  open(config_path, "w"), indent=2)
        print(f"  {config_path} (partial, from best.pt)")


def summarize(run_dir):
    name = os.path.basename(run_dir.rstrip("/"))
    cfg = json.load(open(os.path.join(run_dir, "config.json"))) \
        if os.path.isfile(os.path.join(run_dir, "config.json")) else {}
    test = json.load(open(os.path.join(run_dir, "test.json"))) \
        if os.path.isfile(os.path.join(run_dir, "test.json")) else {}
    ma, args = cfg.get("model_args", {}), cfg.get("args", {})
    arch = [ma.get("variant", "vitt"), ma.get("decoder", "simple")]
    arch += [t for t, on in (("vitskip", ma.get("vit_skips")), ("DS", ma.get("deep_supervision")),
                             ("loraMLP", ma.get("lora_mlp"))) if on]
    row = dict(run=name, seed=args.get("seed", ""), arch="+".join(arch),
               image_size=ma.get("image_size"), encoder_size=ma.get("encoder_size"),
               lora_r=ma.get("lora_r"), lr=args.get("lr", ""),
               small_T=cfg.get("split", {}).get("small_lesion_threshold"),
               best_epoch=test.get("best_epoch"), last_epoch=test.get("last_epoch"),
               best_val_dice=test.get("best_val_dice"), test_dice=test.get("test_dice"))

    rep_path = os.path.join(run_dir, "uncertainty_report.json")
    if os.path.isfile(rep_path):
        rep = json.load(open(rep_path))
        per = rep["per_image"]
        row["mc_test_dice"] = rep.get("mc_mean_dice")
        row["test_dice_median"] = st.median(p["dice"] for p in per)
        row["n_fail_lt0.3"] = sum(p["dice"] < 0.3 for p in per)
        for lab, cls in enumerate(TUMOR_CLASSES):
            ds = [p["dice"] for p in per if p["label"] == lab]
            if ds:
                row[f"dice_{cls.split()[0].lower()}"] = st.mean(ds)
        # Size quartiles by GT area rank: the test split is identical across runs,
        # so quartiles hold the same images regardless of resolution.
        fr = sorted(per, key=lambda p: p["area"])
        q = len(fr) // 4
        for i in range(4):
            chunk = fr[i * q:(i + 1) * q if i < 3 else None]
            row[f"dice_sizeQ{i + 1}"] = st.mean(p["dice"] for p in chunk)
        row["U_small"] = rep["size_stratified"].get("U_small")
        row["U_large"] = rep["size_stratified"].get("U_large")
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(runs_dir()))
    p.add_argument("--backfill", action="store_true")
    args = p.parse_args(argv)

    dirs = sorted(d for d in glob.glob(os.path.join(args.runs, "*")) if os.path.isdir(d))
    if args.backfill:
        for d in dirs:
            backfill(d)
    rows = [summarize(d) for d in dirs]
    keys = list(dict.fromkeys(k for r in rows for k in r))

    with open(os.path.join(args.runs, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows]
    open(os.path.join(args.runs, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # Seed repeats: runs named <config>_s<seed> are grouped with the run <config>.
    groups = {}
    for r in rows:
        groups.setdefault(re.sub(r"_s\d+$", "", r["run"]), []).append(r)
    metrics = ["best_val_dice", "test_dice", "n_fail_lt0.3", "dice_meningioma", "dice_glioma",
               "dice_pituitary", "dice_sizeQ1", "dice_sizeQ2", "dice_sizeQ3", "dice_sizeQ4"]

    def agg(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return ""
        if len(vals) == 1:
            return f"{vals[0]:.4f}"
        return f"{st.mean(vals):.4f} ± {st.stdev(vals):.4f}"

    gmd = ["| config | n_seeds | seeds | " + " | ".join(metrics) + " |",
           "|" + "---|" * (len(metrics) + 3)]
    for g, rs in groups.items():
        gmd.append(f"| {g} | {len(rs)} | {','.join(str(r['seed']) for r in rs)} | "
                   + " | ".join(agg([r.get(m) for r in rs]) for m in metrics) + " |")
    open(os.path.join(args.runs, "summary_by_config.md"), "w").write(
        "mean ± sample std over seeds (same test split)\n\n" + "\n".join(gmd) + "\n")
    print("\n" + "\n".join(gmd))


if __name__ == "__main__":
    main()
