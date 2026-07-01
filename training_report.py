"""
training_report.py
------------------
Print a side-by-side comparison table for depth and segmentation training runs.

Reads real data from:
  - TensorBoard event files  -> train/loss, val/loss, train/grad_norm curves
  - Training log files       -> param counts, dataset sizes, runtime
  - Output directory tree    -> checkpoints, grid images, inference outputs
  - best_model/info.txt      -> best val/loss and the step it came from

Usage:
  # Auto-finds most recent depth + seg run:
  python training_report.py

  # Explicit run directories:
  python training_report.py \\
      --depth outputs/train/depth/runs/2026-07-01/00-41-13 \\
      --seg   outputs/train/seg/runs/2026-07-01/00-47-30

  # Markdown output (default is plain text table):
  python training_report.py --markdown
"""

import argparse
import re
import os
from datetime import datetime
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_latest_run(base: Path) -> Path | None:
    """Return the most recently modified run directory under base."""
    runs = sorted(
        (p for p in base.glob("*/*/") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def _read_tb_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """
    Read TensorBoard event files and return {tag: [(step, value), ...]}.
    Falls back to empty dict if tensorboard is not installed.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    tb_dir = run_dir / "logs" / "tensorboard"
    if not tb_dir.exists():
        return {}

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        scalars[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return scalars


def _parse_log(run_dir: Path) -> dict:
    """
    Parse the plain-text training log for param counts, val/loss lines,
    dataset sizes, best-model events, and runtime.
    """
    result = {
        "mapper_params": None,
        "encoder_params": None,
        "lora_params": None,
        "train_size": None,
        "val_size": None,
        "val_losses": [],      # [(label, val_loss)]  e.g. [("step10", 0.1475)]
        "best_steps": [],      # [(label, val_loss)]  e.g. [("step40", 0.1474)]
        "pipeline": None,
        "schedule": None,
        "start_time": None,
        "end_time": None,
    }

    # Find the log file (depth_training.log or seg_training.log)
    log_files = list(run_dir.glob("*.log"))
    if not log_files:
        return result

    log_path = log_files[0]
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Patterns
    pat_mapper   = re.compile(r"Mapper params.*?([\d,]+)|Number params Mapper.*?([\d,]+)")
    pat_encoder  = re.compile(r"Encoder params.*?([\d,]+)|Number params Encoder.*?([\d,]+)")
    pat_lora     = re.compile(r"LoRA params.*?([\d,]+)|Number params all LoRA.*?([\d,]+)")
    pat_pipeline = re.compile(r"PIPELINE\s*:\s*(.+)")
    pat_schedule = re.compile(r"Schedule\s*:\s*(.+)")
    pat_sizes    = re.compile(r"Train\s*:\s*([\d,]+) images.*?Val: ([\d,]+) images")
    pat_val      = re.compile(r"(?:\[val\]|\[seg val\])\s+(step\d+|epoch\d+):\s*val/loss\s*=\s*([\d.]+)")
    pat_best     = re.compile(r"New best.*?(?:model|segmentation model)\s*[—–-]+\s*(step\d+|epoch\d+),\s*val[/\\]loss=([\d.]+)")
    pat_ts       = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

    timestamps = []
    for line in lines:
        if m := pat_mapper.search(line):
            result["mapper_params"] = (m.group(1) or m.group(2)).replace(",", "")
        if m := pat_encoder.search(line):
            result["encoder_params"] = (m.group(1) or m.group(2)).replace(",", "")
        if m := pat_lora.search(line):
            result["lora_params"] = (m.group(1) or m.group(2)).replace(",", "")
        if m := pat_pipeline.search(line):
            result["pipeline"] = m.group(1).strip()
        if m := pat_schedule.search(line):
            result["schedule"] = m.group(1).strip()
        if m := pat_sizes.search(line):
            result["train_size"] = m.group(1).replace(",", "")
            result["val_size"]   = m.group(2).replace(",", "")
        if m := pat_val.search(line):
            result["val_losses"].append((m.group(1), float(m.group(2))))
        if m := pat_best.search(line):
            result["best_steps"].append((m.group(1), float(m.group(2))))
        if m := pat_ts.search(line):
            timestamps.append(m.group(1))

    if timestamps:
        fmt = "%Y-%m-%d %H:%M:%S"
        result["start_time"] = datetime.strptime(timestamps[0],  fmt)
        result["end_time"]   = datetime.strptime(timestamps[-1], fmt)

    return result


def _read_best_info(run_dir: Path) -> dict:
    """Read best_model/info.txt for best val/loss and source step."""
    info = {"from": None, "val_loss": None}
    info_path = run_dir / "best_model" / "info.txt"
    if not info_path.exists():
        return info
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("from:"):
            info["from"] = line.split(":", 1)[1].strip()
        elif line.startswith("val_loss:"):
            info["val_loss"] = float(line.split(":", 1)[1].strip())
    return info


def _count_checkpoints(run_dir: Path) -> dict:
    """Count step-level checkpoints, grid images, and weight files."""
    ckpt_dirs = [d for d in run_dir.glob("checkpoint-epoch*/*") if d.is_dir()]
    grids = list(run_dir.glob("**/*.jpg"))
    weights = list(run_dir.glob("**/lora-checkpoint.pt"))
    best = run_dir / "best_model"
    return {
        "n_checkpoints": len(ckpt_dirs),
        "n_grids":       len(grids),
        "n_weights":     len(weights),
        "best_exists":   best.exists(),
    }


def _check_inference(run_dir: Path, pipeline_tag: str) -> dict | None:
    """
    Look for inference output for the given pipeline tag (depth or seg).
    pipeline_tag: "depth" or "seg", derived from the training run's tag config.
    """
    repo = run_dir.parents[5]
    inf_base = repo / "outputs" / "inference" / pipeline_tag
    if inf_base.exists():
        grids = list(inf_base.glob("**/*_grid.jpg"))
        if grids:
            return {"tag": pipeline_tag, "n_grids": len(grids), "dir": inf_base}
    return None


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt_num(s: str | None) -> str:
    if s is None:
        return "—"
    n = int(s)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def _fmt_loss(v: float | None) -> str:
    return f"{v:.6f}" if v is not None else "—"


def _fmt_duration(log: dict) -> str:
    if log["start_time"] and log["end_time"]:
        delta = log["end_time"] - log["start_time"]
        mins, secs = divmod(int(delta.total_seconds()), 60)
        return f"{mins}m {secs}s"
    return "—"


# ── main ──────────────────────────────────────────────────────────────────────

def build_run_info(run_dir: Path) -> dict:
    """Collect all stats for one training run."""
    log     = _parse_log(run_dir)
    scalars = _read_tb_scalars(run_dir)
    best    = _read_best_info(run_dir)
    ckpts   = _count_checkpoints(run_dir)
    # Detect pipeline tag from the directory path or log content
    pipeline_str = (log.get("pipeline") or "").lower()
    if "seg" in pipeline_str:
        pipeline_tag = "seg"
    elif "depth" in pipeline_str:
        pipeline_tag = "depth"
    else:
        # Fallback: check the grandparent dir name (outputs/train/{depth|seg}/runs/...)
        pipeline_tag = run_dir.parents[2].name   # parents[2] = "depth" or "seg"
    inf     = _check_inference(run_dir, pipeline_tag)

    # Train/loss: first and last from TensorBoard
    train_loss_curve = scalars.get("train/loss", [])
    first_loss = train_loss_curve[0][1]  if train_loss_curve else None
    last_loss  = train_loss_curve[-1][1] if train_loss_curve else None
    total_steps = train_loss_curve[-1][0] if train_loss_curve else None

    # Grad norm: median of all values (more stable than first/last)
    gnorm_curve = scalars.get("train/grad_norm", [])
    median_gnorm = None
    if gnorm_curve:
        vals = sorted(v for _, v in gnorm_curve)
        median_gnorm = vals[len(vals) // 2]

    # Val/loss samples — take up to 5 evenly spaced
    val_curve = scalars.get("val/loss", [])
    val_samples = []
    if val_curve:
        step = max(1, len(val_curve) // 5)
        indices = list(range(0, len(val_curve), step))[:5]
        val_samples = [(val_curve[i][0], val_curve[i][1]) for i in indices]

    # Fallback to log if TensorBoard not available
    if not val_samples and log["val_losses"]:
        # log has (label, val_loss); convert label "step10" → step=10
        for label, v in log["val_losses"][:5]:
            try:
                step = int(re.search(r"\d+", label).group())
            except Exception:
                step = 0
            val_samples.append((step, v))

    if first_loss is None and last_loss is None:
        # TensorBoard not available; leave as None
        pass

    return {
        "dir":          run_dir,
        "pipeline":     log.get("pipeline") or run_dir.parts[-4] if len(run_dir.parts) >= 4 else "?",
        "schedule":     log.get("schedule") or "—",
        "train_size":   log["train_size"] or "—",
        "val_size":     log["val_size"]   or "—",
        "mapper":       log["mapper_params"],
        "encoder":      log["encoder_params"],
        "lora":         log["lora_params"],
        "first_loss":   first_loss,
        "last_loss":    last_loss,
        "total_steps":  total_steps,
        "median_gnorm": median_gnorm,
        "val_samples":  val_samples,   # [(step, loss), ...]
        "best_loss":    best["val_loss"],
        "best_from":    best["from"],
        "duration":     _fmt_duration(log),
        "n_checkpoints": ckpts["n_checkpoints"],
        "n_grids":       ckpts["n_grids"],
        "n_weights":     ckpts["n_weights"],
        "best_exists":   ckpts["best_exists"],
        "inference":     inf,
    }


def print_report(depth_info: dict, seg_info: dict, markdown: bool = False):
    """Print the comparison table."""

    SEP  = " | " if markdown else "  "
    HEAD = "|" if markdown else ""
    LINE = "|---|---|---|" if markdown else ""

    def row(label, d_val, s_val):
        if markdown:
            print(f"| {label} | {d_val} | {s_val} |")
        else:
            print(f"  {label:<38}  {str(d_val):<28}  {s_val}")

    def section(title):
        if markdown:
            print(f"\n### {title}\n")
            print("| Metric | DEPTH | SEGMENTATION |")
            print("|--------|-------|--------------|")
        else:
            print(f"\n{'-'*80}")
            print(f"  {title}")
            print(f"{'-'*80}")
            print(f"  {'Metric':<38}  {'DEPTH':<28}  SEGMENTATION")
            print(f"  {'-'*38}  {'-'*28}  {'-'*28}")

    print()
    if markdown:
        print("# Depth vs Segmentation — Training Report")
        print(f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
              f"Depth run: `{depth_info['dir'].name}`  |  "
              f"Seg run: `{seg_info['dir'].name}`_\n")
    else:
        print("=" * 80)
        print("  DEPTH vs SEGMENTATION — Training Report")
        print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Depth run : {depth_info['dir']}")
        print(f"  Seg run   : {seg_info['dir']}")
        print("=" * 80)

    # ── Architecture ────────────────────────────────────────────────────────
    section("Architecture")
    row("Pipeline", depth_info["pipeline"] or "DEPTH", seg_info["pipeline"] or "SEGMENTATION")
    row("Mapper params",  _fmt_num(depth_info["mapper"]),  _fmt_num(seg_info["mapper"]))
    row("Encoder params (frozen, 0 grad)", _fmt_num(depth_info["encoder"]), _fmt_num(seg_info["encoder"]))
    row("LoRA params",   _fmt_num(depth_info["lora"]),    _fmt_num(seg_info["lora"]))

    # ── Dataset ─────────────────────────────────────────────────────────────
    section("Dataset")
    row("Train images",  depth_info["train_size"], seg_info["train_size"])
    row("Val images",    depth_info["val_size"],   seg_info["val_size"])
    row("Schedule",      depth_info["schedule"],   seg_info["schedule"])

    # ── Train/loss ──────────────────────────────────────────────────────────
    section("Train Loss  (epsilon-prediction MSE)")
    row("First step loss",  _fmt_loss(depth_info["first_loss"]),  _fmt_loss(seg_info["first_loss"]))
    row("Last step loss",   _fmt_loss(depth_info["last_loss"]),   _fmt_loss(seg_info["last_loss"]))
    row("Total optimizer steps", depth_info["total_steps"] or "—", seg_info["total_steps"] or "—")
    if depth_info["median_gnorm"] is not None or seg_info["median_gnorm"] is not None:
        row("Median grad norm",
            f"{depth_info['median_gnorm']:.4f}" if depth_info["median_gnorm"] else "—",
            f"{seg_info['median_gnorm']:.4f}"   if seg_info["median_gnorm"]   else "—")

    # ── Val/loss ─────────────────────────────────────────────────────────────
    section("Val Loss  (held-out set)")
    # Merge step lists
    d_map = dict(depth_info["val_samples"])
    s_map = dict(seg_info["val_samples"])
    all_steps = sorted(set(d_map) | set(s_map))
    for step in all_steps:
        label = f"val/loss @ step {step}"
        row(label,
            _fmt_loss(d_map.get(step)),
            _fmt_loss(s_map.get(step)))
    row("Best val/loss (best_model/)",
        _fmt_loss(depth_info["best_loss"]),
        _fmt_loss(seg_info["best_loss"]))
    row("Best val/loss saved at",
        depth_info["best_from"] or "—",
        seg_info["best_from"]   or "—")

    # ── Outputs ──────────────────────────────────────────────────────────────
    section("Outputs")
    row("Runtime",          depth_info["duration"],           seg_info["duration"])
    row("Step checkpoints", depth_info["n_checkpoints"],      seg_info["n_checkpoints"])
    row("Weight files",     depth_info["n_weights"],          seg_info["n_weights"])
    row("Grid images",      depth_info["n_grids"],            seg_info["n_grids"])
    row("best_model/ saved", "yes" if depth_info["best_exists"] else "NO",
                              "yes" if seg_info["best_exists"]   else "NO")

    # TensorBoard commands
    if not markdown:
        print()
        print("  TensorBoard:")
        print(f"    tensorboard --logdir \"{depth_info['dir'] / 'logs' / 'tensorboard'}\"")
        print(f"    tensorboard --logdir \"{seg_info['dir']   / 'logs' / 'tensorboard'}\"")
        d_parent = depth_info["dir"].parents[2]
        s_parent = seg_info["dir"].parents[2]
        if d_parent == s_parent:
            print(f"    tensorboard --logdir \"{d_parent}\"  # both together")

    # ── Inference ─────────────────────────────────────────────────────────
    section("Inference")
    def inf_str(info: dict) -> str:
        if info is None:
            return "not run"
        return f"{info['n_grids']} grid(s) in outputs/inference/{info['tag']}/"

    row("Status", inf_str(depth_info["inference"]), inf_str(seg_info["inference"]))

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", type=str, default=None,
                        help="Path to a depth training run dir. Default: most recent.")
    parser.add_argument("--seg",   type=str, default=None,
                        help="Path to a seg training run dir. Default: most recent.")
    parser.add_argument("--markdown", action="store_true",
                        help="Output GitHub-flavoured Markdown instead of plain text.")
    args = parser.parse_args()

    repo = Path(__file__).parent

    # Auto-discover if not specified
    if args.depth:
        depth_dir = Path(args.depth).resolve()
    else:
        depth_dir = _find_latest_run(repo / "outputs" / "train" / "depth" / "runs")
        if depth_dir is None:
            print("[ERROR] No depth training run found under outputs/train/depth/runs/")
            print("  Run: python depth_training.py experiment=train_depth")
            return

    if args.seg:
        seg_dir = Path(args.seg).resolve()
    else:
        seg_dir = _find_latest_run(repo / "outputs" / "train" / "seg" / "runs")
        if seg_dir is None:
            print("[ERROR] No seg training run found under outputs/train/seg/runs/")
            print("  Run: python seg_training.py experiment=train_seg")
            return

    depth_info = build_run_info(depth_dir)
    seg_info   = build_run_info(seg_dir)
    print_report(depth_info, seg_info, markdown=args.markdown)


if __name__ == "__main__":
    main()
