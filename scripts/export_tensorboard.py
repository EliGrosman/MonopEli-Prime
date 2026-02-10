"""Export TensorBoard experiment data to CSV and Markdown for AI analysis.

Reads all experiments from models/ directory, extracts scalar metrics,
and produces:
  1. Per-experiment CSV files (results/csv/<experiment>.csv)
  2. A combined CSV (results/csv/all_experiments.csv)
  3. A comprehensive Markdown report (results/experiment_results.md)

Usage:
    uv run python scripts/export_tensorboard.py [--models-dir models/] [--output-dir results/]
"""

import argparse
import csv
import glob
import os
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_experiment(event_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Load all scalar tags from a TensorBoard event directory."""
    ea = EventAccumulator(event_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data: dict[str, list[tuple[int, float]]] = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data


def find_experiments(models_dir: str) -> dict[str, str]:
    """Find all experiments with TensorBoard data."""
    experiments = {}
    for exp_name in sorted(os.listdir(models_dir)):
        tb_dir = os.path.join(models_dir, exp_name, "tensorboard")
        if not os.path.isdir(tb_dir):
            continue
        event_files = glob.glob(
            os.path.join(tb_dir, "**", "events.out.*"), recursive=True
        )
        if event_files:
            experiments[exp_name] = os.path.dirname(event_files[0])
    return experiments


def write_per_experiment_csv(
    exp_name: str,
    data: dict[str, list[tuple[int, float]]],
    output_dir: str,
) -> str:
    """Write a CSV with all metrics for one experiment. Returns filepath."""
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    filepath = os.path.join(csv_dir, f"{exp_name}.csv")

    # Collect all unique steps across all tags
    all_steps: set[int] = set()
    for points in data.values():
        all_steps.update(step for step, _ in points)
    sorted_steps = sorted(all_steps)

    # Build lookup: tag -> {step: value}
    tag_lookup: dict[str, dict[int, float]] = {}
    for tag, points in data.items():
        tag_lookup[tag] = {step: val for step, val in points}

    tags = sorted(data.keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + tags)
        for step in sorted_steps:
            row = [step] + [
                tag_lookup[tag].get(step, "") for tag in tags  # type: ignore[list-item]
            ]
            writer.writerow(row)

    return filepath


def write_combined_csv(
    all_data: dict[str, dict[str, list[tuple[int, float]]]],
    output_dir: str,
) -> str:
    """Write a single CSV with all experiments, one row per (experiment, step, tag)."""
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    filepath = os.path.join(csv_dir, "all_experiments.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "tag", "step", "value"])
        for exp_name, data in sorted(all_data.items()):
            for tag, points in sorted(data.items()):
                for step, value in points:
                    writer.writerow([exp_name, tag, step, value])

    return filepath


def compute_summary_stats(
    points: list[tuple[int, float]],
) -> dict[str, float | int]:
    """Compute summary statistics for a metric."""
    values = [v for _, v in points]
    if not values:
        return {}
    peak_idx = max(range(len(values)), key=lambda i: values[i])
    last_n = values[-min(5, len(values)) :]
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "final": values[-1],
        "peak_value": values[peak_idx],
        "peak_step": points[peak_idx][0],
        "last_5_mean": statistics.mean(last_n),
        "total_steps": points[-1][0],
        "num_datapoints": len(values),
    }


def format_number(val: float, precision: int = 4) -> str:
    """Format a number nicely."""
    if abs(val) < 0.01 and val != 0:
        return f"{val:.2e}"
    if abs(val) > 1000:
        return f"{val:,.0f}"
    return f"{val:.{precision}f}"


def generate_eval_timeline(
    data: dict[str, list[tuple[int, float]]],
    step_interval: int = 100000,
) -> list[dict[str, str | float]]:
    """Generate a timeline table for eval metrics at regular intervals."""
    vs_random = {s: v for s, v in data.get("eval/vs_random", [])}
    vs_rb = {s: v for s, v in data.get("eval/vs_rule_based", [])}
    lr = {s: v for s, v in data.get("train/learning_rate", [])}

    all_eval_steps = sorted(
        set(list(vs_random.keys()) + list(vs_rb.keys()))
    )
    if not all_eval_steps:
        return []

    max_step = all_eval_steps[-1]
    timeline = []
    for target_step in range(0, max_step + step_interval, step_interval):
        # Find closest actual eval step
        closest = min(all_eval_steps, key=lambda s: abs(s - target_step))
        if abs(closest - target_step) > step_interval * 0.6:
            continue
        row: dict[str, str | float] = {
            "step": f"{closest / 1000:.0f}k",
            "vs_random": f"{vs_random.get(closest, -1) * 100:.0f}%"
            if closest in vs_random
            else "-",
            "vs_rule_based": f"{vs_rb.get(closest, -1) * 100:.0f}%"
            if closest in vs_rb
            else "-",
        }
        if closest in lr:
            row["learning_rate"] = f"{lr[closest]:.2e}"
        timeline.append(row)

    return timeline


def write_markdown_report(
    all_data: dict[str, dict[str, list[tuple[int, float]]]],
    output_dir: str,
) -> str:
    """Write a comprehensive Markdown report for AI consumption."""
    filepath = os.path.join(output_dir, "experiment_results.md")

    lines: list[str] = []
    lines.append("# Training Experiment Results")
    lines.append("")
    lines.append(f"**Generated**: {__import__('datetime').date.today()}")
    lines.append(f"**Experiments**: {len(all_data)}")
    lines.append(
        f"**Source**: TensorBoard event files from `models/` directory"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== Executive Summary ==========
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "| Experiment | Total Steps | Peak vs Random | Peak vs Rule-Based |"
        " Final vs Random | Final vs Rule-Based | Status |"
    )
    lines.append("|" + "---|" * 7)

    for exp_name, data in sorted(all_data.items()):
        vs_random = data.get("eval/vs_random", [])
        vs_rb = data.get("eval/vs_rule_based", [])

        random_stats = compute_summary_stats(vs_random) if vs_random else {}
        rb_stats = compute_summary_stats(vs_rb) if vs_rb else {}

        peak_r = (
            f"{random_stats.get('peak_value', 0) * 100:.0f}%"
            f" @ {random_stats.get('peak_step', 0) / 1000:.0f}k"
            if random_stats
            else "-"
        )
        peak_rb = (
            f"{rb_stats.get('peak_value', 0) * 100:.0f}%"
            f" @ {rb_stats.get('peak_step', 0) / 1000:.0f}k"
            if rb_stats
            else "-"
        )
        final_r = (
            f"{random_stats.get('final', 0) * 100:.0f}%"
            if random_stats
            else "-"
        )
        final_rb = (
            f"{rb_stats.get('final', 0) * 100:.0f}%"
            if rb_stats
            else "-"
        )
        total = (
            f"{random_stats.get('total_steps', 0) / 1000:.0f}k"
            if random_stats
            else "-"
        )

        # Determine status
        final_r_val = random_stats.get("final", 0)
        peak_r_val = random_stats.get("peak_value", 0)
        if final_r_val == 0 and peak_r_val > 0.2:
            status = "Collapsed"
        elif final_r_val >= peak_r_val * 0.8:
            status = "Stable"
        elif final_r_val >= peak_r_val * 0.5:
            status = "Declining"
        else:
            status = "Degraded"

        lines.append(
            f"| {exp_name} | {total} | {peak_r} | {peak_rb} |"
            f" {final_r} | {final_rb} | {status} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== Per-Experiment Detail ==========
    for exp_name, data in sorted(all_data.items()):
        lines.append(f"## {exp_name}")
        lines.append("")

        # Key eval metrics
        key_tags = ["eval/vs_random", "eval/vs_rule_based"]
        lines.append("### Evaluation Metrics")
        lines.append("")
        for tag in key_tags:
            if tag not in data:
                continue
            stats = compute_summary_stats(data[tag])
            lines.append(f"**{tag}**:")
            lines.append(
                f"- Peak: {stats['peak_value'] * 100:.1f}%"
                f" at step {stats['peak_step']:,}"
            )
            lines.append(f"- Final: {stats['final'] * 100:.1f}%")
            lines.append(
                f"- Last 5 evals mean: {stats['last_5_mean'] * 100:.1f}%"
            )
            lines.append(f"- Overall mean: {stats['mean'] * 100:.1f}%")
            lines.append("")

        # Timeline table
        timeline = generate_eval_timeline(data)
        if timeline:
            lines.append("### Eval Timeline")
            lines.append("")
            headers = list(timeline[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "---|" * len(headers))
            for row in timeline:
                lines.append(
                    "| " + " | ".join(str(row.get(h, "-")) for h in headers) + " |"
                )
            lines.append("")

        # Training diagnostics
        lines.append("### Training Diagnostics")
        lines.append("")
        diag_tags = [
            "train/value_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/learning_rate",
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "time/fps",
        ]
        lines.append("| Metric | Min | Max | Mean | Final |")
        lines.append("|---|---|---|---|---|")
        for tag in diag_tags:
            if tag not in data:
                continue
            stats = compute_summary_stats(data[tag])
            lines.append(
                f"| {tag} | {format_number(stats['min'])} |"
                f" {format_number(stats['max'])} |"
                f" {format_number(stats['mean'])} |"
                f" {format_number(stats['final'])} |"
            )
        lines.append("")

        # Raw eval data dump (every data point for the key metrics)
        lines.append("### Raw Eval Data")
        lines.append("")
        lines.append("```")
        lines.append("step,vs_random,vs_rule_based")
        vs_r_dict = {s: v for s, v in data.get("eval/vs_random", [])}
        vs_rb_dict = {s: v for s, v in data.get("eval/vs_rule_based", [])}
        all_eval = sorted(set(list(vs_r_dict.keys()) + list(vs_rb_dict.keys())))
        for step in all_eval:
            r = f"{vs_r_dict[step]:.4f}" if step in vs_r_dict else ""
            rb = f"{vs_rb_dict[step]:.4f}" if step in vs_rb_dict else ""
            lines.append(f"{step},{r},{rb}")
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ========== Cross-Experiment Comparison ==========
    lines.append("## Cross-Experiment Comparison")
    lines.append("")
    lines.append("### Peak Performance")
    lines.append("")
    lines.append("| Experiment | Peak vs Random | Peak vs Rule-Based | Steps to Peak (RB) |")
    lines.append("|---|---|---|---|")
    for exp_name, data in sorted(all_data.items()):
        vs_r = data.get("eval/vs_random", [])
        vs_rb = data.get("eval/vs_rule_based", [])
        r_stats = compute_summary_stats(vs_r) if vs_r else {}
        rb_stats = compute_summary_stats(vs_rb) if vs_rb else {}
        lines.append(
            f"| {exp_name}"
            f" | {r_stats.get('peak_value', 0) * 100:.1f}%"
            f" | {rb_stats.get('peak_value', 0) * 100:.1f}%"
            f" | {rb_stats.get('peak_step', 0):,} |"
        )
    lines.append("")

    lines.append("### Stability (Last 5 Evals)")
    lines.append("")
    lines.append("| Experiment | Last 5 Mean vs Random | Last 5 Mean vs RB | Collapsed? |")
    lines.append("|---|---|---|---|")
    for exp_name, data in sorted(all_data.items()):
        vs_r = data.get("eval/vs_random", [])
        vs_rb = data.get("eval/vs_rule_based", [])
        r_stats = compute_summary_stats(vs_r) if vs_r else {}
        rb_stats = compute_summary_stats(vs_rb) if vs_rb else {}
        collapsed = (
            "YES"
            if r_stats.get("final", 0) == 0 and r_stats.get("peak_value", 0) > 0.2
            else "No"
        )
        lines.append(
            f"| {exp_name}"
            f" | {r_stats.get('last_5_mean', 0) * 100:.1f}%"
            f" | {rb_stats.get('last_5_mean', 0) * 100:.1f}%"
            f" | {collapsed} |"
        )
    lines.append("")

    # ========== Data Files ==========
    lines.append("---")
    lines.append("")
    lines.append("## Data Files")
    lines.append("")
    lines.append("For detailed analysis, the raw data is available as CSV:")
    lines.append("")
    lines.append("- `results/csv/all_experiments.csv` - All experiments in long format (experiment, tag, step, value)")
    lines.append("- `results/csv/<experiment>.csv` - Per-experiment wide format (step, metric1, metric2, ...)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Auto-generated by `scripts/export_tensorboard.py`*")

    report = "\n".join(lines)
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(report)

    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard data to CSV + Markdown"
    )
    parser.add_argument(
        "--models-dir",
        default="models/",
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Output directory for CSV and Markdown files",
    )
    args = parser.parse_args()

    print(f"Scanning {args.models_dir} for experiments...")
    experiments = find_experiments(args.models_dir)

    if not experiments:
        print("No experiments found!")
        return

    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}")

    all_data: dict[str, dict[str, list[tuple[int, float]]]] = {}

    for exp_name, event_dir in experiments.items():
        print(f"\nLoading {exp_name}...")
        data = load_experiment(event_dir)
        all_data[exp_name] = data

        csv_path = write_per_experiment_csv(exp_name, data, args.output_dir)
        n_tags = len(data)
        n_points = sum(len(v) for v in data.values())
        print(f"  {n_tags} metrics, {n_points} total datapoints -> {csv_path}")

    combined_path = write_combined_csv(all_data, args.output_dir)
    print(f"\nCombined CSV -> {combined_path}")

    report_path = write_markdown_report(all_data, args.output_dir)
    print(f"Markdown report -> {report_path}")

    print("\nDone! Files ready for AI analysis:")
    print(f"  - {report_path} (structured analysis)")
    print(f"  - {combined_path} (raw data)")


if __name__ == "__main__":
    main()
