#!/usr/bin/env python3
"""
Check whether any AE feature's winsorize bound lands outside post_scaling_clip
once scaled, i.e. whether legitimate values get saturated to the clip ceiling.

Usage: python scripts/check_feature_saturation.py [path/to/full_config.json]
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_path",
        nargs="?",
        default="exports/full_config.json",
        help="Path to the exported config JSON (default: exports/full_config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    with config_path.open() as f:
        config = json.load(f)

    clip_params = config["preprocessing"]["ae_clip_params"]
    scaler = config["preprocessing"]["ae_scaler"]
    feature_names = scaler["feature_names"]
    means = scaler["mean"]
    stds = scaler["std"]
    post_min = config["preprocessing"]["post_scaling_clip"]["min"]
    post_max = config["preprocessing"]["post_scaling_clip"]["max"]

    rows = []
    for i, name in enumerate(feature_names):
        mean, std = means[i], stds[i]
        lower = clip_params[name]["lower"]
        upper = clip_params[name]["upper"]
        z_lower = (lower - mean) / std
        z_upper = (upper - mean) / std
        saturates = z_upper > post_max or z_lower < post_min
        rows.append((name, z_lower, z_upper, saturates))

    # Worst offenders first.
    rows.sort(key=lambda r: -max(abs(r[1]), abs(r[2])))

    print(f"post_scaling_clip: [{post_min}, {post_max}]\n")
    print(f"{'Feature':<22}{'z_lower':>10}{'z_upper':>10}   Saturates?")
    print("-" * 55)
    for name, z_lower, z_upper, saturates in rows:
        flag = "<-- YES" if saturates else ""
        print(f"{name:<22}{z_lower:>10.3f}{z_upper:>10.3f}   {flag}")

    n_saturating = sum(1 for *_, saturates in rows if saturates)
    print(f"\n{n_saturating}/{len(rows)} features saturate at the post-scaling clip.")


if __name__ == "__main__":
    main()
