"""Check the staged Habibi checkpoint against the Tunisian project vocabulary.

Run without arguments from anywhere in the repository::

    python src/f5_tts/scripts/verify_habibi_checkpoint.py
"""

import argparse
from pathlib import Path

from cached_path import cached_path
from safetensors import safe_open


HABIBI_MSA_CHECKPOINT = "hf://SWivid/Habibi-TTS/Specialized/MSA/model_200000.safetensors"
HABIBI_MSA_VOCAB = "hf://SWivid/Habibi-TTS/Specialized/MSA/vocab.txt"


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def find_checkpoint(repo_root: Path) -> Path:
    expected_dir = repo_root / "ckpts" / "F5TTS_v1_Base_TUN_vocos_char_TunisianTTS"
    checkpoints = sorted(expected_dir.glob("*.safetensors"))
    if len(checkpoints) == 1:
        return checkpoints[0]
    if len(checkpoints) > 1:
        locations = "\n".join(f"  - {path}" for path in checkpoints) or "  (none found)"
        raise FileNotFoundError(
            "Expected exactly one .safetensors checkpoint in the Tunisian checkpoint folder. Found:\n" + locations
        )

    # This checks the Hugging Face cache first and only downloads if the model
    # is not present locally, matching the Gradio fine-tuning workflow.
    return Path(cached_path(HABIBI_MSA_CHECKPOINT))


def read_vocab_tokens(vocab_path: Path) -> list[str]:
    """Read tokens exactly as F5-TTS's char tokenizer does."""
    with vocab_path.open("r", encoding="utf-8") as vocab_file:
        return [line[:-1] for line in vocab_file]


def first_difference(left: list[str], right: list[str]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return len(left) if len(left) != len(right) else None


def main() -> None:
    parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint_path = find_checkpoint(repo_root)
    vocab_path = repo_root / "data" / "TunisianTTS_char" / "vocab.txt"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Project vocabulary not found: {vocab_path}")

    tokens = read_vocab_tokens(vocab_path)
    vocab_rows = len(tokens)
    vocab_size = len(set(tokens))
    duplicate_count = vocab_rows - vocab_size
    base_vocab_path = Path(cached_path(HABIBI_MSA_VOCAB))
    base_tokens = read_vocab_tokens(base_vocab_path)

    print(f"checkpoint: {checkpoint_path}")
    print(f"vocab: {vocab_path}")
    print(f"base vocab: {base_vocab_path}")
    print(f"vocabulary rows: {vocab_rows}")
    print(f"effective tokenizer entries: {vocab_size}")
    print(f"duplicate vocabulary entries: {duplicate_count}")
    print(f"cached MSA vocabulary rows: {len(base_tokens)}")
    difference = first_difference(tokens, base_tokens)
    if difference is None:
        print("project vocabulary matches the cached MSA vocabulary exactly.")
    else:
        project_token = repr(tokens[difference]) if difference < len(tokens) else "<missing>"
        base_token = repr(base_tokens[difference]) if difference < len(base_tokens) else "<missing>"
        print(
            f"first difference from the cached MSA vocabulary: index {difference}; "
            f"project={project_token}, base={base_token}"
        )

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        embedding_keys = [key for key in checkpoint.keys() if key.endswith("text_embed.text_embed.weight")]
        if not embedding_keys:
            raise ValueError("No text embedding tensor found in checkpoint")
        shapes = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in embedding_keys}

    expected_embedding_rows = len(base_tokens) + 1
    mismatches = {key: shape for key, shape in shapes.items() if shape[0] != expected_embedding_rows}
    for key, shape in shapes.items():
        print(f"{key}: {shape}")
    print(f"checkpoint embedding rows: {next(iter(shapes.values()))[0]}")
    print(f"expected Habibi embedding rows: {expected_embedding_rows}")
    if mismatches:
        raise ValueError(f"Embedding/official-vocabulary row mismatch: {mismatches}")
    if difference is not None:
        raise ValueError("Project vocabulary differs from the cached official MSA vocabulary; restore it before training.")
    print("Checkpoint text embeddings and vocabulary are compatible.")

    

if __name__ == "__main__":
    main()
