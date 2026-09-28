#!/usr/bin/env python3


import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

from components import DataPreprocess, DeepAutoencoder, Exporter
from model import download_dataset, get_dataset_config, list_available_datasets
from utils import LauncherWizard, Logger, pipeline_stage

if __name__ == "__main__":
    Logger.setup_logging()

    log = Logger("Main")

    total_start = time.perf_counter()

    available = list_available_datasets()

    parser = argparse.ArgumentParser(
        description="Mantis ML Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Single dataset (auto-download from Kaggle)
  python main.py -s xxx/xxx-dataset -a

  # Merge multiple datasets
  python main.py -s xxx/xxx-dataset,yyy/yyy-dataset -a

  # All three datasets
  python main.py -s xxx/xxx-dataset,yyy/yyy-dataset,zzz/zzz-dataset -a

  # Use local dataset paths (override auto-download)
  python main.py -s xxx/xxx-dataset --path "/data/xxx" -a

  # Run individual stages
  python main.py -s xxx/xxx-dataset -dp     # Preprocess only
  python main.py -da                        # Train LSTM Autoencoder
  python main.py -ep                        # Export to ONNX
  python main.py --view-plots               # Browse the latest run's plots

  # Pretrain on one dataset, then fine-tune on another
  python main.py -s xxx/xxx-dataset -dp -da  # pretrain -> ./artifacts
  python main.py -s yyy/yyy-dataset -dp -da --finetune

Available datasets: {', '.join(available)}
        """,
    )

    parser.add_argument(
        "-s",
        "--set",
        help=f"Dataset kaggle id(s), comma-separated ({', '.join(available)})",
    )
    parser.add_argument(
        "--path",
        help="Optional: path(s) to local dataset directories, comma-separated. "
        "If omitted, datasets are auto-downloaded via kagglehub.",
    )

    parser.add_argument(
        "-a", "--all", action="store_true", help="Run complete pipeline"
    )
    parser.add_argument(
        "-dp", "--datapreprocess", action="store_true", help="Data preprocessing"
    )
    parser.add_argument(
        "-da",
        "--deepautoencoder",
        action="store_true",
        help="Train LSTM Deep Autoencoder",
    )
    parser.add_argument(
        "-ep", "--export", action="store_true", help="Export models to ONNX"
    )

    parser.add_argument(
        "--view-plots",
        action="store_true",
        help="Browse the latest training run's plots in the terminal",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from checkpoint (e.g. ./artifacts/autoencoder_temp-v14.ckpt)",
    )
    parser.add_argument(
        "--finetune",
        nargs="?",
        const="./artifacts",
        default=None,
        metavar="DIR",
        help="Fine-tune from a pretrained model directory containing "
        "deep_autoencoder.pt + deep_ae_config.pkl (default: ./artifacts). "
        "Reuses its preprocessing and architecture, trains at a reduced lr.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="With --finetune/--resume: freeze the encoder, train only the decoder",
    )

    args = parser.parse_args()
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    results_plots = []
    results_summary = None

    if not any(
        [
            args.all,
            args.datapreprocess,
            args.deepautoencoder,
            args.export,
            args.view_plots,
        ]
    ):
        if interactive:
            log.info("No stage flags given — launching interactive launcher...")
            selection = LauncherWizard().run()
            if not selection:
                parser.print_help()
                sys.exit(0)
            args.datapreprocess = selection["datapreprocess"]
            args.deepautoencoder = selection["deepautoencoder"]
            args.export = selection["export"]
            args.set = selection["set"]
            args.path = selection["path"]
            args.finetune = selection["finetune"]
            args.freeze_encoder = selection["freeze_encoder"]
        else:
            parser.print_help()
            sys.exit(0)

    if args.resume and args.finetune:
        parser.error("--resume and --finetune can't be used together")
    if args.freeze_encoder and not (args.finetune or args.resume):
        parser.error("--freeze-encoder needs --finetune or --resume")

    if args.all or args.datapreprocess:
        data_preproces_start = time.perf_counter()

        if not args.set:
            parser.error("-s/--set is required for preprocessing")

        dataset_names = [s.strip() for s in args.set.split(",")]
        dataset_configs = [get_dataset_config(name) for name in dataset_names]

        if args.path:
            dataset_paths = [p.strip() for p in args.path.split(",")]
            if len(dataset_names) != len(dataset_paths):
                parser.error(
                    f"Number of datasets ({len(dataset_names)}) "
                    f"must match number of paths ({len(dataset_paths)})"
                )
        else:
            log.info("No --path provided, downloading datasets via kagglehub...")
            dataset_paths = [download_dataset(cfg) for cfg in dataset_configs]

        log.info(f"Datasets: {', '.join(dataset_names)}")

        with pipeline_stage("Data Preprocessing"):
            with DataPreprocess(dataset_configs, dataset_paths) as dp:
                dp.load_datasets()
                dp.statistics_dataset()
                dp.feature_preparation()
                dp.output_result()

        data_preproces_end = time.perf_counter()
        log.info(
            f"Data preprocessing execution time: {timedelta(seconds=(data_preproces_end - data_preproces_start))}"
        )

    if args.all or args.deepautoencoder:
        deep_autoencoder_start = time.perf_counter()

        with pipeline_stage("LSTM Deep Autoencoder"):
            with DeepAutoencoder(
                resume_ckpt=args.resume,
                finetune_from=args.finetune,
                freeze_encoder=args.freeze_encoder,
            ) as da:
                da.check_environment()
                da.load_previous_artifacts()
                da.load_data()
                da.prepare_data()
                da.preprocess_data()
                da.build_sequences()
                da.build_autoencoder()
                da.train_autoencoder()
                da.predict_autoencoder()
                da.bootstrap_metrics()
                da.save_results()
                da.generate_visualizations()
                results_plots = list(da.plot_paths)
                results_summary = da.results_summary()

        deep_autoencoder_end = time.perf_counter()
        log.info(
            f"LSTM Deep Autoencoder execution time: {timedelta(seconds=(deep_autoencoder_end - deep_autoencoder_start))}"
        )

    if args.all or args.export:
        export_start = time.perf_counter()

        with pipeline_stage("Export models to ONNX", delay=0):
            with Exporter() as ep:
                ep.load_models()
                ep.export_deep_ae_onnx()
                ep.build_config_json()
                ep.save_config_json()
                ep.verify_onnx_models()
                ep.verify_onnx_export()
                ep.print_summary()

        export_end = time.perf_counter()
        log.info(
            f"Export models to ONNX execution time: {timedelta(seconds=(export_end - export_start))}"
        )

    total_end = time.perf_counter()

    log.info(f"Execution time: {timedelta(seconds=(total_end - total_start))}")

    if args.view_plots and not interactive:
        log.warning("--view-plots needs an interactive terminal (docker run -it)")
    elif interactive and (results_plots or args.view_plots):
        # Lazy import: textual-image probes the terminal's graphics support
        # at import time, which only makes sense right before showing it.
        from utils.results_tui import ResultsViewer, latest_plot_set

        results_plots = results_plots or latest_plot_set(Path("plots"))
        if results_plots:
            ResultsViewer(results_plots, summary=results_summary).run()
        else:
            log.warning("--view-plots: no training plots found in ./plots")
