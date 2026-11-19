#!/usr/bin/env python3

import argparse
import dataclasses
import sys
from typing import List

import ujson

from model import PipelineError, ReportError, VerdictReportConfig
from pipeline import (
    ArgsParser,
    CsvPool,
    PipelinePlanner,
    PipelineRequest,
    Stage,
)
from report import VerdictReport
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
  ./main.py --verdict-report
  ./main.py --verdict-report --window-secs 300 --top 20 --json

Stages read CSV files from $DATASET_ROOT (default ./datasets).
--args takes section.key=value pairs. Sections: dp (PreprocessConfig),
da (DeepAutoencoderConfig), ex (ExportConfig) and run (resume, model_dir).
The section prefix is required on every key.

--verdict-report runs no stage: it reads the ML Verdict column of every CSV in
$DATASET_ROOT and reports how many rows dp.exclude_verdicts drops and what
excluding the sources it flags would cost, without changing anything.
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
    report = parser.add_argument_group("verdict report")
    report.add_argument(
        "--verdict-report",
        action="store_true",
        help="Report ML Verdict counts per source IP instead of running stages",
    )
    report.add_argument(
        "--trigger-verdicts",
        default=",".join(VerdictReportConfig.trigger_verdicts),
        metavar="V1,V2",
        help="Verdicts that flag a source (default: %(default)s)",
    )
    report.add_argument(
        "--window-secs",
        type=int,
        default=VerdictReportConfig.window_secs,
        help="Half-width of the window around each trigger row (default: %(default)s)",
    )
    report.add_argument(
        "--top",
        type=int,
        default=VerdictReportConfig.top,
        help="Flagged sources to list (default: %(default)s)",
    )
    report.add_argument("--json", action="store_true", help="Print the report as JSON")
    return parser


def run_verdict_report(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    config = VerdictReportConfig(
        trigger_verdicts=tuple(
            v.strip() for v in args.trigger_verdicts.split(",") if v.strip()
        ),
        window_secs=args.window_secs,
        top=args.top,
    )
    try:
        report = VerdictReport(config)
    except ReportError as exc:
        parser.error(str(exc))
    pool = CsvPool(CsvPool.default_data_root())
    files = pool.files()
    if not files:
        parser.error(f"No CSV files in {pool.root}")
    result = report.build(files, pool.root)
    if args.json:
        sys.stdout.write(ujson.dumps(dataclasses.asdict(result), indent=2) + "\n")
    else:
        sys.stdout.write(VerdictReport.render(result) + "\n")


def main() -> None:
    Logger.setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    stages: List[Stage] = args.stages or []
    if args.verdict_report:
        if stages:
            parser.error("--verdict-report runs no stage; drop the stage flags")
        run_verdict_report(parser, args)
        return
    if not stages:
        parser.print_help()
        return
    planner = PipelinePlanner(
        CsvPool(CsvPool.default_data_root()), ArgsParser(), accept_path=None
    )
    try:
        options = planner.plan(PipelineRequest(stages=stages, args=args.args))
    except PipelineError as exc:
        parser.error(str(exc))
    # Imported here so --verdict-report needs no training dependencies.
    from pipeline.runner import PipelineRunner

    PipelineRunner().run(options)


if __name__ == "__main__":
    main()
