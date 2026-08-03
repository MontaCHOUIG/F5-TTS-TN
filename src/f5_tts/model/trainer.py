from __future__ import annotations

import gc
import math
import os
import time
from collections import deque

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists


# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        mlflow_tracker=None,
        mlflow_log_every_updates: int = 10,
        inference_callback=None,
        inference_every_updates: int = 0,
    ):
        # DDP normally keeps a second gradient-sized communication bucket.  Make
        # gradients views into that bucket to avoid an extra full-model buffer on
        # each GPU when training through ``accelerate launch``.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, gradient_as_bucket_view=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                    "bnb_optimizer": bnb_optimizer,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = None
            if self.accelerator.is_main_process:
                self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor
        self.mlflow_tracker = mlflow_tracker
        self.mlflow_log_every_updates = max(1, int(mlflow_log_every_updates))
        self.inference_callback = inference_callback
        self.inference_every_updates = max(0, int(inference_every_updates))

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, fused=True)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                checkpoint_file = f"{self.checkpoint_path}/model_last.pt"
                self.accelerator.save(checkpoint, checkpoint_file)
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                checkpoint_file = f"{self.checkpoint_path}/model_{update}.pt"
                self.accelerator.save(checkpoint, checkpoint_file)
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")
            if self.mlflow_tracker is not None:
                self.mlflow_tracker.log_checkpoint(checkpoint_file, step=update, is_best=False)

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update
        training_started_at = time.monotonic()
        epoch_durations = deque(maxlen=5)
        metric_window_started_at = training_started_at
        updates_since_log = 0
        samples_since_log = 0
        frames_since_log = 0
        best_train_loss = float("inf")
        loss_window = deque(maxlen=self.mlflow_log_every_updates)
        accumulated_samples = 0
        accumulated_frames = 0
        accumulated_loss = 0.0
        accumulation_microbatches = 0

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            epoch_started_at = time.monotonic()
            epoch_loss_sum = 0.0
            epoch_updates = 0
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                grad_norm = None
                duration_loss = None
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    accumulated_samples += len(text_inputs)
                    accumulated_frames += int(mel_lengths.sum().item())

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        duration_loss = dur_loss.item()
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
                    accumulated_loss += loss.detach().item()
                    accumulation_microbatches += 1
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    elif self.accelerator.sync_gradients and self.mlflow_tracker is not None and self.is_main:
                        parameter_norms = [
                            parameter.grad.detach().norm(2)
                            for parameter in self.model.parameters()
                            if parameter.grad is not None
                        ]
                        if parameter_norms:
                            grad_norm = torch.stack(parameter_norms).norm(2)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    mean_loss = self.accelerator.reduce(
                        torch.tensor(
                            accumulated_loss / max(accumulation_microbatches, 1), device=self.accelerator.device
                        ),
                        reduction="mean",
                    ).item()
                    global_samples = int(
                        self.accelerator.reduce(
                            torch.tensor(accumulated_samples, device=self.accelerator.device), reduction="sum"
                        ).item()
                    )
                    global_frames = int(
                        self.accelerator.reduce(
                            torch.tensor(accumulated_frames, device=self.accelerator.device), reduction="sum"
                        ).item()
                    )
                    accumulated_samples = 0
                    accumulated_frames = 0
                    accumulated_loss = 0.0
                    accumulation_microbatches = 0

                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    updates_since_log += 1
                    samples_since_log += global_samples
                    frames_since_log += global_frames
                    loss_window.append(mean_loss)
                    best_train_loss = min(best_train_loss, mean_loss)
                    epoch_loss_sum += mean_loss
                    epoch_updates += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=mean_loss)

                    if self.mlflow_tracker is not None and self.is_main:
                        applied_learning_rate = float(self.optimizer.param_groups[0]["lr"])
                        metrics = {
                            "train/loss": mean_loss,
                            "train/learning_rate": applied_learning_rate,
                            "train/learning_rate_ratio": applied_learning_rate / self.learning_rate,
                            "scheduler/warmup_progress": min(global_update / max(warmup_updates, 1), 1.0),
                            "scheduler/decay_progress": min(
                                max(global_update - warmup_updates, 0) / max(decay_updates, 1), 1.0
                            ),
                            "progress/epoch": epoch + 1,
                            "batch/global_samples": global_samples,
                            "batch/global_frames": global_frames,
                            "batch/mean_frames_per_sample": global_frames / max(global_samples, 1),
                        }
                        if grad_norm is not None:
                            metrics["train/gradient_norm"] = grad_norm
                        if duration_loss is not None:
                            metrics["train/duration_loss"] = duration_loss
                        if global_update % self.mlflow_log_every_updates == 0:
                            now = time.monotonic()
                            window_seconds = max(now - metric_window_started_at, 1e-9)
                            metrics.update(
                                {
                                    "train/loss_window_mean": sum(loss_window) / len(loss_window),
                                    "train/loss_best": best_train_loss,
                                    "throughput/steps_per_second": updates_since_log / window_seconds,
                                    "throughput/samples_per_second": samples_since_log / window_seconds,
                                    "throughput/frames_per_second": frames_since_log / window_seconds,
                                    "time/elapsed_seconds": now - training_started_at,
                                }
                            )
                            metric_window_started_at = now
                            updates_since_log = 0
                            samples_since_log = 0
                            frames_since_log = 0
                        self.mlflow_tracker.log_metrics(metrics, step=global_update)

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                if self.logger == "tensorboard" and self.accelerator.is_main_process:
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode(), self.accelerator.autocast():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0, :, :ref_audio_len].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

                should_run_inference = (
                    self.inference_callback is not None
                    and self.inference_every_updates > 0
                    and self.accelerator.sync_gradients
                    and global_update % self.inference_every_updates == 0
                )
                if should_run_inference:
                    # Keep all DDP ranks at the same update while rank zero runs
                    # the optional, memory-intensive inference job.
                    self.accelerator.wait_for_everyone()
                    if self.is_main:
                        try:
                            inference_artifacts = self.inference_callback(
                                self.accelerator.unwrap_model(self.model),
                                global_update,
                                self.accelerator,
                            )
                            if self.mlflow_tracker is not None:
                                for artifact in inference_artifacts:
                                    self.mlflow_tracker.log_audio_sample(
                                        artifact["audio_path"], global_update, artifact["sample_rate"]
                                    )
                                    self.mlflow_tracker.log_mel_spectrogram(artifact["spectrogram_path"], global_update)
                                self.mlflow_tracker.log_metrics(
                                    {"inference/generated_samples": len(inference_artifacts)}, step=global_update
                                )
                        except Exception as error:
                            print(f"Optional MLflow inference warning at update {global_update}: {error}")
                            if self.mlflow_tracker is not None:
                                self.mlflow_tracker.log_metrics({"inference/failures": 1}, step=global_update)
                        finally:
                            self.model.train()
                    self.accelerator.wait_for_everyone()

            epoch_seconds = time.monotonic() - epoch_started_at
            epoch_durations.append(epoch_seconds)
            cumulative_seconds = time.monotonic() - training_started_at
            recent_average = sum(epoch_durations) / len(epoch_durations)
            remaining_epochs = max(self.epochs - (epoch + 1), 0)
            if self.mlflow_tracker is not None and self.is_main:
                self.mlflow_tracker.log_epoch_time(epoch + 1, epoch_seconds)
                self.mlflow_tracker.log_metrics(
                    {
                        "train/epoch_loss": epoch_loss_sum / max(epoch_updates, 1),
                        "train/epoch_updates": epoch_updates,
                        "time/cumulative_seconds": cumulative_seconds,
                        "time/eta_seconds": recent_average * remaining_epochs,
                        "progress/epoch": epoch + 1,
                    },
                    step=global_update,
                )

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
