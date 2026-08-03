"""Failure-tolerant MLflow tracking for F5-TTS fine-tuning.

MLflow is imported lazily so normal F5-TTS training does not require the
tracking dependencies. Every public method catches tracking errors: an MLflow
server or artifact failure must never stop model training.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class MLflowTracker:
    """Small, reusable wrapper around MLflow's run and logging APIs."""

    def __init__(
        self,
        tracking_uri: str = "file:./mlruns",
        enabled: bool = False,
        log_system_metrics: bool = True,
        log_checkpoint_artifacts: bool = True,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.enabled = enabled
        self.log_system_metrics = log_system_metrics
        self.log_checkpoint_artifacts = log_checkpoint_artifacts
        self.mlflow = None
        self.run = None
        self._manual_system_metrics = False
        self._warned_messages: set[str] = set()
        self._nvml = None
        self._nvml_handle = None

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.mlflow is not None and self.run is not None)

    def _warn(self, operation: str, error: Exception | str) -> None:
        message = f"MLflow tracking warning ({operation}): {error}"
        if message not in self._warned_messages:
            print(message)
            self._warned_messages.add(message)

    def _import_mlflow(self) -> bool:
        if self.mlflow is not None:
            return True
        try:
            import mlflow

            self.mlflow = mlflow
            return True
        except Exception as error:
            self._warn("import", error)
            return False

    @staticmethod
    def _to_dict(config: Any) -> dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(config):
                return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
        except Exception:
            pass
        if hasattr(config, "__dict__"):
            return vars(config)
        return {"config": config}

    @classmethod
    def _flatten_params(cls, config: Any, prefix: str = "") -> dict[str, str | int | float | bool]:
        flattened: dict[str, str | int | float | bool] = {}
        for key, value in cls._to_dict(config).items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                flattened.update(cls._flatten_params(value, full_key))
                continue
            if isinstance(value, (list, tuple, set)):
                value = json.dumps(list(value), ensure_ascii=False, default=str)
            elif value is None:
                value = "null"
            elif not isinstance(value, (str, int, float, bool)):
                value = str(value)
            flattened[full_key[:250]] = value if not isinstance(value, str) else value[:500]
        return flattened

    @staticmethod
    def _git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    def _enable_system_metrics(self) -> None:
        if not self.log_system_metrics or self.mlflow is None:
            return
        enable_builtin = getattr(self.mlflow, "enable_system_metrics_logging", None)
        if callable(enable_builtin):
            try:
                enable_builtin()
                return
            except Exception as error:
                self._warn("enable system metrics", error)
        self._manual_system_metrics = True
        self._initialize_nvml()

    def _initialize_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as error:
            self._warn("initialize GPU metrics fallback", error)

    def start_run(
        self,
        config: Any,
        run_name: str | None = None,
        experiment_name: str = "f5-tts-finetune",
    ) -> None:
        if not self.enabled or not self._import_mlflow():
            return
        try:
            timestamp = datetime.now(timezone.utc)
            effective_name = run_name or f"f5-tts-{timestamp.strftime('%Y%m%d-%H%M%S')}"
            self.mlflow.set_tracking_uri(self.tracking_uri)
            self.mlflow.set_experiment(experiment_name)
            self._enable_system_metrics()
            self.run = self.mlflow.start_run(run_name=effective_name)
            tags = {
                "f5_tts.run_started_utc": timestamp.isoformat(),
                "f5_tts.tracking_uri": self.tracking_uri,
            }
            git_commit = self._git_commit()
            if git_commit:
                tags["git.commit"] = git_commit
            self.mlflow.set_tags(tags)
            self.log_params(config)
            print(f"MLflow run started: {effective_name} ({self.run.info.run_id})")
        except Exception as error:
            self._warn("start run", error)
            self.run = None

    def log_params(self, config_dict: Any) -> None:
        if not self.active:
            return
        try:
            params = self._flatten_params(config_dict)
            if params:
                self.mlflow.log_params(params)
        except Exception as error:
            self._warn("log params", error)

    @staticmethod
    def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
        numeric: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                if hasattr(value, "item"):
                    value = value.item()
                numeric[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return numeric

    def _collect_manual_system_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            import psutil

            metrics["system/cpu_utilization_percent"] = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            metrics["system/memory_utilization_percent"] = memory.percent
            metrics["system/memory_used_mb"] = memory.used / (1024**2)
        except Exception as error:
            self._warn("collect CPU metrics fallback", error)

        if self._nvml is not None and self._nvml_handle is not None:
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                metrics["system/gpu_utilization_percent"] = utilization.gpu
                metrics["system/gpu_memory_utilization_percent"] = memory.used / memory.total * 100
                metrics["system/gpu_memory_used_mb"] = memory.used / (1024**2)
            except Exception as error:
                self._warn("collect GPU metrics fallback", error)
        return metrics

    def log_metrics(self, metrics_dict: Mapping[str, Any], step: int) -> None:
        if not self.active:
            return
        try:
            metrics = self._numeric_metrics(metrics_dict)
            if self._manual_system_metrics:
                metrics.update(self._collect_manual_system_metrics())
            if metrics:
                self.mlflow.log_metrics(metrics, step=int(step))
        except Exception as error:
            self._warn("log metrics", error)

    def log_epoch_time(self, epoch: int, seconds: float) -> None:
        self.log_metrics({"time/epoch_seconds": seconds, "progress/epoch": epoch}, step=epoch)

    def log_audio_sample(self, filepath_or_array: Any, step: int, sample_rate: int) -> None:
        if not self.active:
            return
        try:
            artifact_path = f"inference/update_{int(step)}"
            if isinstance(filepath_or_array, (str, os.PathLike)):
                self.mlflow.log_artifact(str(filepath_or_array), artifact_path=artifact_path)
                return

            audio = np.asarray(filepath_or_array, dtype=np.float32).squeeze()
            log_audio = getattr(self.mlflow, "log_audio", None)
            if callable(log_audio):
                log_audio(audio, sample_rate, f"{artifact_path}/generated.wav")
                return

            import soundfile as sf

            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = Path(temp_dir) / "generated.wav"
                sf.write(audio_path, audio, sample_rate)
                self.mlflow.log_artifact(str(audio_path), artifact_path=artifact_path)
        except Exception as error:
            self._warn("log audio sample", error)

    def log_mel_spectrogram(self, image_or_array: Any, step: int) -> None:
        if not self.active:
            return
        try:
            artifact_path = f"inference/update_{int(step)}"
            if isinstance(image_or_array, (str, os.PathLike)):
                self.mlflow.log_artifact(str(image_or_array), artifact_path=artifact_path)
                return

            import matplotlib.pyplot as plt

            image = np.asarray(image_or_array).squeeze()
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / "mel_spectrogram.png"
                figure, axis = plt.subplots(figsize=(12, 4))
                plotted = axis.imshow(image, origin="lower", aspect="auto", interpolation="none")
                axis.set_title(f"Generated mel-spectrogram at update {step}")
                axis.set_xlabel("Frame")
                axis.set_ylabel("Mel channel")
                figure.colorbar(plotted, ax=axis)
                figure.tight_layout()
                figure.savefig(image_path, dpi=150)
                plt.close(figure)
                self.mlflow.log_artifact(str(image_path), artifact_path=artifact_path)
        except Exception as error:
            self._warn("log mel-spectrogram", error)

    def log_checkpoint(self, path: str | os.PathLike[str], step: int, is_best: bool = False) -> None:
        if not self.active:
            return
        try:
            checkpoint_path = Path(path).resolve()
            metadata = {
                "path": str(checkpoint_path),
                "step": int(step),
                "is_best": bool(is_best),
            }
            self.mlflow.set_tag("checkpoint.latest_path", str(checkpoint_path))
            self.mlflow.set_tag("checkpoint.latest_step", str(step))
            if is_best:
                self.mlflow.set_tag("checkpoint.best_path", str(checkpoint_path))
            if self.log_checkpoint_artifacts and checkpoint_path.is_file():
                self.mlflow.log_artifact(str(checkpoint_path), artifact_path=f"checkpoints/update_{int(step)}")
            log_dict = getattr(self.mlflow, "log_dict", None)
            if callable(log_dict):
                log_dict(metadata, f"checkpoints/update_{int(step)}/metadata.json")
        except Exception as error:
            self._warn("log checkpoint", error)

    def end_run(self) -> None:
        if self.mlflow is None or self.run is None:
            return
        try:
            self.mlflow.end_run()
        except Exception as error:
            self._warn("end run", error)
        finally:
            self.run = None
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception:
                    pass
