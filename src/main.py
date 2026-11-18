#!/usr/bin/env python3

import argparse
from typing import List

from pipeline import (
    ArgsParser,
    CsvPool,
    PipelineError,
    PipelinePlanner,
    PipelineRequest,
    Stage,
)
from pipeline.runner import PipelineRunner
from utils import Logger

EPILOG = """
Examples:
  ./main.py -a
  ./main.py -dp --args "dp.exclude_verdicts=none"
  ./main.py -da --args "da.epochs=50 da.batch_size=4096"
  ./main.py -da --args "run.resume=artifacts/autoencoder_temp-v14.ckpt"
  ./main.py -ft --args "run.model_dir=./old_model da.finetune_lr_scale=0.05"
  ./main.py -sc --args "da.analysis_threshold=p99.5"
  ./main.py -fa --args "da.analysis_threshold=p95"
  ./main.py -ex

Stages read CSV files from $DATASET_ROOT (default ./datasets).
--args takes section.key=value pairs. Sections: dp (PreprocessConfig),
da (DeepAutoencoderConfig), ex (ExportConfig) and run (resume, model_dir).
The section prefix is required on every key.
"""

STAGE_FLAGS = (
    ("-a", "--all", Stage.ALL, "Run dp + da + ex"),
    ("-dp", "--datapreprocess", Stage.DP, "Preprocess every CSV in $DATASET_ROOT"),
    ("-da", "--deepautoencoder", Stage.DA, "Train LSTM Autoencoder"),
    ("-ft", "--finetune", Stage.FT, "Finetune the model in model_dir on loaded data"),
    ("-sc", "--score", Stage.SC, "Score preprocessed data with the model in model_dir"),
    ("-fa", "--feature-analysis", Stage.FA, "Per-feature error analysis of the model"),
    ("-ex", "--export", Stage.EX, "Export model to ONNX"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mantis ML Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    for short, long, stage, text in STAGE_FLAGS:
        parser.add_argument(
            short,
            long,
            dest="stages",
            action="append_const",
            const=stage,
            help=text,
        )
    parser.add_argument(
        "--args",
        default="",
        metavar="SECTION.KEY=VALUE ...",
        help='Extra parameters, e.g. --args "da.epochs=50 da.clip_min=-4"',
    )
    return parser


def main() -> None:
    Logger.setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    stages: List[Stage] = args.stages or []
    if not stages:
        parser.print_help()
        return
    planner = PipelinePlanner(CsvPool(CsvPool.default_data_root()), ArgsParser(), accept_path=None)
    try:
        options = planner.plan(PipelineRequest(stages=stages, args=args.args))
    except PipelineError as exc:
        parser.error(str(exc))
    PipelineRunner().run(options)


if __name__ == "__main__":
    main()
