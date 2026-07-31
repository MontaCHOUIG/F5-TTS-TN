"""Verify and copy the cached Habibi checkpoint for the project vocabulary.

DiT allocates one reserved filler embedding in addition to the vocabulary
entries. The compatible checkpoint therefore has ``len(vocab) + 1`` rows;
this script preserves all rows and never changes the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cached_path import cached_path
from safetensors import safe_open


HABIBI_MSA_CHECKPOINT = "hf://SWivid/Habibi-TTS/Specialized/MSA/model_200000.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint path (default: ckpts/F5TTS_v1_Base_TUN_vocos_char_TunisianTTS/pretrained_msa_project_vocab_compatible.safetensors)",
    )
    return parser.parse_args()


def read_tokens(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as vocab_file:
        return [line[:-1] for line in vocab_file]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    source_checkpoint = Path(cached_path(HABIBI_MSA_CHECKPOINT))
    project_vocab = repo_root / "data" / "TunisianTTS_char" / "vocab.txt"
    output_path = args.output or (
        repo_root
        / "ckpts"
        / "F5TTS_v1_Base_TUN_vocos_char_TunisianTTS"
        / "pretrained_msa_project_vocab_compatible.safetensors"
    )
    output_path = output_path.expanduser().resolve()

    if not project_vocab.is_file():
        raise FileNotFoundError(f"Project vocabulary not found: {project_vocab}")
    project_tokens = read_tokens(project_vocab)
    project_vocab_size = len(set(project_tokens))
    if not project_tokens or project_tokens[0] != " ":
        raise ValueError("Project vocabulary token 0 must be exactly one space for the F5-TTS char tokenizer")
    if project_vocab_size != len(project_tokens):
        raise ValueError("Project vocabulary contains duplicate tokens; tokenizer indices would be ambiguous")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output checkpoint: {output_path}")

    with safe_open(source_checkpoint, framework="pt", device="cpu") as checkpoint:
        embedding_keys = [key for key in checkpoint.keys() if key.endswith("text_embed.text_embed.weight")]
        shapes = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in embedding_keys}
    if not embedding_keys:
        raise ValueError("No text embedding tensor found in checkpoint")
    expected_embedding_rows = project_vocab_size + 1  # +1 for DiT's reserved filler embedding
    invalid = {key: shape for key, shape in shapes.items() if shape[0] != expected_embedding_rows}
    if invalid:
        raise ValueError(
            f"The project vocabulary requires {expected_embedding_rows} embedding rows (including DiT's filler row), "
            f"but the source checkpoint has: {invalid}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, output_path)
    print(f"Source checkpoint: {source_checkpoint}")
    print(f"Project vocabulary: {project_vocab} ({project_vocab_size} entries)")
    print(f"Created vocabulary-compatible checkpoint: {output_path}")


if __name__ == "__main__":
    main()
