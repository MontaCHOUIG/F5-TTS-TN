"""Estimate epochs required to reach a target number of optimizer updates."""

import argparse
import math

from torch.utils.data import SequentialSampler

from f5_tts.model.dataset import DynamicBatchSampler, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="TunisianTTS")
    parser.add_argument("--tokenizer", default="char")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch-size-per-gpu", type=int, default=9600)
    parser.add_argument("--max-samples-per-gpu", type=int, default=64)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=200000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_values = (
        args.gpus,
        args.batch_size_per_gpu,
        args.max_samples_per_gpu,
        args.grad_accumulation_steps,
        args.max_updates,
    )
    if any(value < 1 for value in positive_values):
        raise ValueError("GPU, batch, accumulation, sample, and update values must all be positive")

    train_dataset = load_dataset(args.dataset, args.tokenizer)
    sampler = SequentialSampler(train_dataset)
    batch_sampler = DynamicBatchSampler(
        sampler,
        args.batch_size_per_gpu,
        max_samples=args.max_samples_per_gpu,
        random_seed=666,
        drop_residual=False,
    )

    batches_per_epoch = len(batch_sampler)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gpus / args.grad_accumulation_steps)
    epochs = math.ceil(args.max_updates / updates_per_epoch)
    print(f"batches/epoch: {batches_per_epoch}")
    print(f"optimizer updates/epoch: {updates_per_epoch}")
    print(f"epochs for {args.max_updates} updates: {epochs}")


if __name__ == "__main__":
    main()
