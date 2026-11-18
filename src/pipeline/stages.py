from typing import Dict, Protocol

from components import DataPreprocess, DeepAutoencoder, Exporter
from model import get_dataset_config

from .csv_pool import SOURCE_NAME
from .options import PipelineOptions
from .stage import Stage


class PipelineStage(Protocol):
    title: str
    delay: float

    def run(self, options: PipelineOptions) -> None: ...


class DataPreprocessStage:
    title = "Data Preprocessing"
    delay = 3.0

    def run(self, options: PipelineOptions) -> None:
        configs = [get_dataset_config(SOURCE_NAME)]
        with DataPreprocess(configs, [options.data_root], options.preprocess) as dp:
            dp.load_datasets()
            dp.statistics_dataset()
            dp.feature_preparation()
            dp.output_result()


class DeepAutoencoderStage:
    title = "LSTM Deep Autoencoder"
    delay = 3.0

    def run(self, options: PipelineOptions) -> None:
        with DeepAutoencoder(options.autoencoder) as da:
            da.check_environment()
            da.load_data()
            da.prepare_data()
            da.preprocess_data()
            da.build_sequences()
            da.build_autoencoder()
            da.train_autoencoder(resume_ckpt=options.resume)
            da.predict_autoencoder()
            da.analyze_feature_errors()
            da.bootstrap_metrics()
            da.save_results()
            da.generate_visualizations()


class FinetuneStage:
    title = "LSTM Deep Autoencoder Finetune"
    delay = 3.0

    def run(self, options: PipelineOptions) -> None:
        with DeepAutoencoder(options.autoencoder) as da:
            da.check_environment()
            da.load_data()
            da.load_pretrained(options.model_dir)
            da.prepare_data()
            da.preprocess_data()
            da.build_sequences()
            da.build_autoencoder()
            da.finetune_baseline()
            da.train_autoencoder()
            da.predict_autoencoder()
            da.finetune_report()
            da.analyze_feature_errors()
            da.bootstrap_metrics()
            da.save_results()
            da.generate_visualizations()


class ScoreStage:
    title = "Score"
    delay = 3.0

    def run(self, options: PipelineOptions) -> None:
        with DeepAutoencoder(options.autoencoder) as da:
            da.check_environment()
            da.load_data()
            da.load_for_scoring(options.model_dir)
            da.prepare_data()
            da.preprocess_data()
            da.build_autoencoder()
            da.score_report()
            da.analyze_feature_errors()


class FeatureAnalysisStage:
    title = "Feature Error Analysis"
    delay = 3.0

    def run(self, options: PipelineOptions) -> None:
        with DeepAutoencoder(options.autoencoder) as da:
            da.check_environment()
            da.load_data()
            da.match_trained_split()
            da.prepare_data()
            da.preprocess_data()
            da.load_trained_model()
            da.analyze_feature_errors()


class ExportStage:
    title = "Export models to ONNX"
    delay = 0.0

    def run(self, options: PipelineOptions) -> None:
        with Exporter(options.export) as ep:
            ep.load_models()
            ep.export_deep_ae_onnx()
            ep.build_config_json()
            ep.save_config_json()
            ep.verify_onnx_models()
            ep.verify_onnx_export()
            ep.print_summary()


STAGES: Dict[Stage, PipelineStage] = {
    Stage.DP: DataPreprocessStage(),
    Stage.DA: DeepAutoencoderStage(),
    Stage.FT: FinetuneStage(),
    Stage.SC: ScoreStage(),
    Stage.FA: FeatureAnalysisStage(),
    Stage.EX: ExportStage(),
}
