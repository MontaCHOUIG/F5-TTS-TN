"""F5-TTS fine-tuning entry point with optional MLflow tracking.

This keeps the normal Hydra/Accelerate training path intact. It does not create
or evaluate a validation split. Optional inference uses ``voice.wav`` and one
generation prompt per non-empty line of ``test.txt`` from a user-managed input
directory (``data/Tn_inference`` by default).
"""

from __future__ import annotations

import json
import os
import gc
from glob import glob
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from safetensors import safe_open

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer, seed_everything
from f5_tts.train.mlflow_tracker import MLflowTracker


os.chdir(str(files("f5_tts").joinpath("../..")))  # local editable install: use the repository root


class FixedInferenceSampler:
    """Generate comparable samples from user-managed reference inputs."""

    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        mel_spec_type: str,
        is_local_vocoder: bool = False,
        local_vocoder_path: str | None = None,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 1.0,
        seed: int = 666,
        release_vocoder_after_run: bool = True,
    ) -> None:
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.mel_spec_type = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path or ""
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.speed = speed
        self.seed = seed
        self.release_vocoder_after_run = release_vocoder_after_run
        self.vocoder = None
        self.ref_audio: str | None = None
        self.ref_text: str | None = None
        self._voice_signature: tuple[int, int] | None = None

    @property
    def voice_path(self) -> Path:
        return self.input_directory / "voice.wav"

    @property
    def text_path(self) -> Path:
        return self.input_directory / "test.txt"

    def _read_prompts(self) -> list[str]:
        if not self.text_path.is_file():
            raise FileNotFoundError(f"MLflow inference prompt file not found: {self.text_path}")
        prompts = [line.strip() for line in self.text_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"MLflow inference prompt file contains no non-empty rows: {self.text_path}")
        return prompts

    def _prepare_reference(self) -> None:
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        if not self.voice_path.is_file():
            raise FileNotFoundError(f"MLflow inference reference audio not found: {self.voice_path}")
        stat = self.voice_path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature != self._voice_signature:
            self.ref_audio, self.ref_text = preprocess_ref_audio_text(str(self.voice_path), "")
            self._voice_signature = signature

    def _prepare_vocoder(self, device) -> None:
        from f5_tts.infer.utils_infer import load_vocoder

        if self.vocoder is None:
            self.vocoder = load_vocoder(
                vocoder_name=self.mel_spec_type,
                is_local=self.is_local_vocoder,
                local_path=self.local_vocoder_path,
                device=device,
            )

    def __call__(self, model: CFM, update: int, accelerator) -> list[dict[str, object]]:
        import soundfile as sf
        import torch

        from f5_tts.infer.utils_infer import infer_process, save_spectrogram

        prompts = self._read_prompts()
        self._prepare_reference()
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        self._prepare_vocoder(accelerator.device)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        artifacts: list[dict[str, object]] = []
        try:
            model.eval()
            for index, prompt in enumerate(prompts, start=1):
                try:
                    seed_everything(self.seed + index)
                    with torch.inference_mode(), accelerator.autocast():
                        waveform, sample_rate, spectrogram = infer_process(
                            self.ref_audio,
                            self.ref_text,
                            prompt,
                            model,
                            self.vocoder,
                            mel_spec_type=self.mel_spec_type,
                            progress=None,
                            nfe_step=self.nfe_step,
                            cfg_strength=self.cfg_strength,
                            sway_sampling_coef=self.sway_sampling_coef,
                            speed=self.speed,
                            device=accelerator.device,
                        )
                    if waveform is None or spectrogram is None:
                        raise RuntimeError("F5-TTS returned no generated audio")

                    stem = f"update_{update}_prompt_{index:03d}"
                    audio_path = self.output_directory / f"{stem}.wav"
                    spectrogram_path = self.output_directory / f"{stem}_mel.png"
                    sf.write(audio_path, waveform, sample_rate)
                    save_spectrogram(spectrogram, spectrogram_path)
                    artifacts.append(
                        {
                            "audio_path": str(audio_path),
                            "spectrogram_path": str(spectrogram_path),
                            "sample_rate": sample_rate,
                        }
                    )
                except Exception as error:
                    print(f"Optional inference warning for test.txt row {index}: {error}")
        finally:
            if self.release_vocoder_after_run and self.vocoder is not None:
                self.vocoder.to("cpu")
                self.vocoder = None
                gc.collect()
                if accelerator.device.type == "cuda":
                    torch.cuda.empty_cache()
        return artifacts


def _build_tunisian_audit(
    model_cfg, checkpoint_path: str, tokenizer: str, tokenizer_path: str, vocab_size: int
) -> None:
    with open(
        str(files("f5_tts").joinpath(f"../../data/{model_cfg.datasets.name}_{tokenizer}/vocab.txt")),
        "r",
        encoding="utf-8",
    ) as vocab_file:
        vocab_rows = sum(1 for _ in vocab_file)
    checkpoint_candidates = sorted(glob(os.path.join(checkpoint_path, "pretrained_*.safetensors")))
    checkpoint_shapes = {}
    for candidate in checkpoint_candidates:
        with safe_open(candidate, framework="pt", device="cpu") as checkpoint:
            checkpoint_shapes[candidate] = {
                key: list(checkpoint.get_slice(key).get_shape())
                for key in checkpoint.keys()
                if key.endswith("text_embed.text_embed.weight")
            }
    audit = {
        "event": "trainer_tokenizer_audit",
        "checkpoint_path": checkpoint_path,
        "checkpoint_candidates": checkpoint_candidates,
        "checkpoint_embedding_shapes": checkpoint_shapes,
        "dataset": model_cfg.datasets.name,
        "tokenizer": tokenizer,
        "tokenizer_path_argument": str(tokenizer_path),
        "vocab_rows": vocab_rows,
        "effective_vocab_size": vocab_size,
        "resolved_model_config": OmegaConf.to_container(model_cfg.model, resolve=True),
    }
    os.makedirs(checkpoint_path, exist_ok=True)
    audit_path = os.path.join(checkpoint_path, "habibi_trainer_audit.jsonl")
    with open(audit_path, "a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n")
    print("HABIBI TRAINER AUDIT: " + json.dumps(audit, ensure_ascii=False, sort_keys=True))


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg) -> None:
    # This entry point never generates samples from a training batch.
    model_cfg.ckpts.log_samples = False
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    wandb_project = model_cfg.ckpts.get("wandb_project", "CFM-TTS")
    wandb_run_name = model_cfg.ckpts.get(
        "wandb_run_name",
        f"{model_cfg.model.name}_{mel_spec_type}_{tokenizer}_{model_cfg.datasets.name}",
    )
    wandb_resume_id = model_cfg.ckpts.get("wandb_resume_id", None)

    tokenizer_path = model_cfg.datasets.name if tokenizer != "custom" else model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    checkpoint_path = str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}"))

    if model_cfg.model.name == "F5TTS_v1_Base_TUN":
        _build_tunisian_audit(model_cfg, checkpoint_path, tokenizer, tokenizer_path, vocab_size)

    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    if model_cfg.model.name == "F5TTS_v1_Base_TUN":
        model_embedding_rows = model.transformer.text_embed.text_embed.weight.shape[0]
        model_audit = {
            "event": "trainer_model_audit",
            "effective_vocab_size": vocab_size,
            "model_embedding_rows": model_embedding_rows,
            "reserved_filler_rows": 1,
            "vocab_size_offset": model_cfg.model.get("vocab_size_offset", None),
        }
        audit_path = os.path.join(checkpoint_path, "habibi_trainer_audit.jsonl")
        with open(audit_path, "a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(model_audit, ensure_ascii=False, sort_keys=True) + "\n")
        print("HABIBI MODEL AUDIT: " + json.dumps(model_audit, ensure_ascii=False, sort_keys=True))
        if model_embedding_rows != vocab_size + 1:
            raise RuntimeError(f"Tokenizer/model embedding mismatch before checkpoint loading: {model_audit}")

    mlflow_cfg = model_cfg.get("mlflow", {})
    use_mlflow = bool(mlflow_cfg.get("enabled", False))
    tracker = MLflowTracker(
        tracking_uri=str(mlflow_cfg.get("tracking_uri", "file:./mlruns")),
        enabled=use_mlflow,
        log_system_metrics=bool(mlflow_cfg.get("log_system_metrics", True)),
        log_checkpoint_artifacts=bool(mlflow_cfg.get("log_checkpoint_artifacts", True)),
    )

    inference_cfg = mlflow_cfg.get("inference", {})
    use_inference = use_mlflow and bool(inference_cfg.get("enabled", False))
    inference_callback = None
    inference_every_updates = 0
    if use_inference:
        inference_every_updates = int(inference_cfg.get("every_updates", model_cfg.ckpts.save_per_updates))
        inference_callback = FixedInferenceSampler(
            input_directory=str(inference_cfg.get("directory", "data/Tn_inference")),
            output_directory=os.path.join(checkpoint_path, "mlflow_inference"),
            mel_spec_type=mel_spec_type,
            is_local_vocoder=bool(model_cfg.model.vocoder.is_local),
            local_vocoder_path=model_cfg.model.vocoder.local_path,
            nfe_step=int(inference_cfg.get("nfe_step", 32)),
            cfg_strength=float(inference_cfg.get("cfg_strength", 2.0)),
            sway_sampling_coef=float(inference_cfg.get("sway_sampling_coef", -1.0)),
            speed=float(inference_cfg.get("speed", 1.0)),
            seed=int(inference_cfg.get("seed", 666)),
            release_vocoder_after_run=bool(inference_cfg.get("release_vocoder_after_run", True)),
        )

    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        # Never sample from a training batch in this entry point. Optional samples
        # come exclusively from the user-managed MLflow inference directory.
        log_samples=False,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        mlflow_tracker=tracker if use_mlflow else None,
        mlflow_log_every_updates=int(mlflow_cfg.get("log_every_updates", 10)),
        inference_callback=inference_callback,
        inference_every_updates=inference_every_updates,
    )

    if trainer.is_main:
        tracker.start_run(
            OmegaConf.to_container(model_cfg, resolve=True),
            run_name=mlflow_cfg.get("run_name", None),
            experiment_name=str(mlflow_cfg.get("experiment_name", "f5-tts-finetune")),
        )

    try:
        train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)
        trainer.train(
            train_dataset,
            num_workers=model_cfg.datasets.num_workers,
            resumable_with_seed=666,
        )
    finally:
        if trainer.is_main:
            tracker.end_run()


if __name__ == "__main__":
    main()
