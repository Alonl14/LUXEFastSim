"""Assemble a 3-stage BED comparison (current vs +EMA vs +EMA+calibration).

The pure core takes a bed_fn(stage) -> float so it can be tested without data.
On the cluster, wrap utils.get_batch_ed_histograms / utils.get_ed in bed_fn.
"""


def three_stage_table(bed_fn, stages=("current", "+ema", "+ema+calib")):
    return {stage: bed_fn(stage) for stage in stages}


def format_table(results):
    lines = ["BED comparison (lower is better):"]
    for stage, value in results.items():
        lines.append(f"  {stage:<14} {value:.6g}")
    return "\n".join(lines)
